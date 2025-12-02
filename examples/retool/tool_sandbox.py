"""
Tool sandbox module for safe code execution and tool management.

This module provides:
- PythonSandbox: Safe Python code execution environment
- ToolRegistry: Tool registration and execution management
- Memory management utilities
"""

import asyncio
import gc
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Any

import psutil

# Configuration for tool execution
TOOL_CONFIGS = {
    "max_turns": 16,
    "max_tool_calls": 16,
    "tool_concurrency": 32,  # Aggressive: 32 concurrent processes
    # Python interpreter settings
    "python_timeout": 120,  # 2 minutes for complex calculations
    "python_memory_limit": "4GB",  # 4GB per Python process
    "python_cpu_limit": 1,
    # Memory management settings
    "max_memory_usage": 12288,  # 12GB total (75% of 16GB)
    "cleanup_threshold": 6144,  # 6GB
    "aggressive_cleanup_threshold": 3072,  # 3GB
    "force_cleanup_threshold": 9216,  # 9GB
}

# Global semaphore for controlling concurrent tool executions
SEMAPHORE = asyncio.Semaphore(TOOL_CONFIGS["tool_concurrency"])


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()
    if hasattr(gc, "collect"):
        gc.collect()


def aggressive_cleanup_memory():
    """More aggressive memory cleanup"""
    # Force multiple garbage collection cycles
    for _ in range(3):
        gc.collect()

    # Clear Python's internal caches
    import sys

    # Note: sys.intern doesn't have a clear method, so we skip this
    # Clear module cache if possible
    if hasattr(sys, "modules"):
        # Don't clear all modules, but clear some common ones that might cache data
        modules_to_clear = ["numpy", "pandas", "matplotlib", "scipy"]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                if hasattr(module, "clear_cache"):
                    module.clear_cache()


def check_and_cleanup_memory():
    """Check memory usage and perform appropriate cleanup"""
    current_memory = get_memory_usage()

    if current_memory > TOOL_CONFIGS["force_cleanup_threshold"]:
        # Force aggressive cleanup
        aggressive_cleanup_memory()
        return f"Warning: High memory usage ({current_memory:.1f}MB), performed aggressive cleanup"
    elif current_memory > TOOL_CONFIGS["cleanup_threshold"]:
        # Normal cleanup
        cleanup_memory()
        return f"Info: Memory usage ({current_memory:.1f}MB), performed cleanup"
    elif current_memory > TOOL_CONFIGS["aggressive_cleanup_threshold"]:
        # Light cleanup
        gc.collect()
        return f"Info: Memory usage ({current_memory:.1f}MB), performed light cleanup"

    return None


class PythonSandbox:
    """Python code sandbox, provides safe code execution environment"""

    def __init__(self, timeout: int = 10, memory_limit: str = "100MB"):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.allowed_modules = {
            "math",
            "random",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "operator",
            "statistics",
            "decimal",
            "fractions",
        }

    def _check_code_safety(self, code: str) -> tuple[bool, str]:
        """Check code safety by scanning for dangerous patterns"""
        # Check for dangerous operations
        dangerous_patterns = [
            r"import\s+os",
            r"import\s+sys",
            r"import\s+subprocess",
            r"import\s+shutil",
            r"import\s+glob",
            r"import\s+pathlib",
            r"__import__",
            r"eval\s*\(",
            r"exec\s*\(",
            r"open\s*\(",
            r"file\s*\(",
            r"input\s*\(",
            r"raw_input\s*\(",
            r"compile\s*\(",
            r"execfile\s*\(",
            r"getattr\s*\(",
            r"setattr\s*\(",
            r"delattr\s*\(",
            r"hasattr\s*\(",
            r"globals\s*\(",
            r"locals\s*\(",
            r"vars\s*\(",
            r"dir\s*\(",
            r"type\s*\(",
            r"isinstance\s*\(",
            r"issubclass\s*\(",
            r"super\s*\(",
            r"property\s*\(",
            r"staticmethod\s*\(",
            r"classmethod\s*\(",
            r"__\w+__",  # double underscore methods
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Code contains dangerous pattern: {pattern}"

        # Check imported modules
        import_pattern = r"import\s+(\w+)"
        from_pattern = r"from\s+(\w+)"

        imports = re.findall(import_pattern, code)
        froms = re.findall(from_pattern, code)

        all_imports = set(imports + froms)
        for imp in all_imports:
            if imp not in self.allowed_modules:
                return False, f"Import of '{imp}' is not allowed"

        return True, "Code is safe"

    @contextmanager
    def _create_safe_environment(self):
        """Create safe execution environment with temporary directory"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="python_sandbox_")

        try:
            # Create safe Python script
            script_path = os.path.join(temp_dir, "code.py")

            # Set environment variables
            env = os.environ.copy()
            env["PYTHONPATH"] = temp_dir
            env["PYTHONUNBUFFERED"] = "1"

            yield script_path, env, temp_dir

        finally:
            # Clean up temporary directory
            try:
                import shutil

                shutil.rmtree(temp_dir)
            except Exception:
                pass

    async def execute_code(self, code: str) -> str:
        """Execute Python code in sandbox with safety checks"""
        # Check memory usage before execution
        current_memory = get_memory_usage()
        if current_memory > TOOL_CONFIGS["max_memory_usage"]:
            aggressive_cleanup_memory()
            return "Error: Memory usage too high, please try again"

        # Check code safety
        is_safe, message = self._check_code_safety(code)
        if not is_safe:
            return f"Error: {message}"

        # Add necessary wrapper code with memory limits
        # Properly indent the user code within the try block
        # Handle indentation properly by adding 4 spaces to each line
        indented_code = "\n".join("    " + line for line in code.split("\n"))

        wrapped_code = f"""import sys
import traceback
from io import StringIO
import resource

# Set memory limit (4GB)
try:
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, -1))
except:
    pass

# Redirect stdout and stderr
old_stdout = sys.stdout
old_stderr = sys.stderr
stdout_capture = StringIO()
stderr_capture = StringIO()
sys.stdout = stdout_capture
sys.stderr = stderr_capture

try:
    # User code
{indented_code}
    
    # Get output
    stdout_output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()
    
    # Restore standard output
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Return result
    result = ""
    if stdout_output:
        result += f"Output:\\n{{stdout_output}}"
    if stderr_output:
        result += f"\\nErrors:\\n{{stderr_output}}"
    
    print(result)
    
except Exception as e:
    # Restore standard output
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Return error information
    error_msg = f"Error: {{str(e)}}\\nTraceback:\\n{{traceback.format_exc()}}"
    print(error_msg)"""

        with self._create_safe_environment() as (script_path, env, temp_dir):
            # Write code to file
            with open(script_path, "w") as f:
                f.write(wrapped_code)

            try:
                # Use subprocess to run code
                process = subprocess.Popen(
                    ["python3", script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=temp_dir,
                    text=True,
                )

                # Set timeout
                try:
                    stdout, stderr = process.communicate(timeout=self.timeout)

                    if process.returncode == 0:
                        result = stdout.strip()
                    else:
                        result = f"Error: Process exited with code " f"{process.returncode}\n{stderr}"

                except subprocess.TimeoutExpired:
                    process.kill()
                    result = f"Error: Code execution timed out after " f"{self.timeout} seconds"

            except Exception as e:
                result = f"Error: Failed to execute code: {str(e)}"

            # Check memory usage after execution and cleanup if needed
            cleanup_message = check_and_cleanup_memory()
            if cleanup_message:
                print(f"Memory cleanup: {cleanup_message}")

            return result


class ToolRegistry:
    """Tool registry, manages available tools and their execution"""

    def __init__(self):
        self.tools = {}
        self.python_sandbox = PythonSandbox(
            timeout=TOOL_CONFIGS["python_timeout"], memory_limit=TOOL_CONFIGS["python_memory_limit"]
        )
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools in the registry"""
        # Python code interpreter
        self.register_tool(
            "code_interpreter",
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "description": "A tool for executing Python code in a safe sandbox environment.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string", "description": "The Python code to execute"}},
                        "required": ["code"],
                    },
                },
            },
        )

    def register_tool(self, name: str, tool_spec: dict[str, Any]):
        """Register a new tool in the registry"""
        self.tools[name] = tool_spec

    def get_tool_specs(self) -> list[dict[str, Any]]:
        """Get all tool specifications as a list"""
        return list(self.tools.values())

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call with the given arguments"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"

        async with SEMAPHORE:
            if tool_name == "code_interpreter":
                return await self._execute_python(arguments)
            else:
                return f"Error: Tool '{tool_name}' not implemented"

    async def _execute_python(self, arguments: dict[str, Any]) -> str:
        """Execute Python code using the sandbox"""
        code = arguments.get("code", "")
        if not code.strip():
            return "Error: No code provided"

        # Execute code in sandbox
        result = await self.python_sandbox.execute_code(code)
        return result


# Global tool registry instance
tool_registry = ToolRegistry()
