#!/usr/bin/env python3
"""
Debug test script for OpenAI Tool Call Adapter

This script demonstrates the debug output from the OpenAI adapter
"""

from openai_tool_adapter import create_openai_adapter


def test_debug_output():
    """
    Test the debug output functionality
    """
    print("=" * 60)
    print("OpenAI Tool Call Adapter Debug Test")
    print("=" * 60)
    
    # Example tool information
    tools_info = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search web information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    # Create adapter
    print("\n1. Creating OpenAI adapter...")
    adapter = create_openai_adapter(tools_info, "qwen25")

    # Test 1: Response with tool call
    print("\n2. Testing response with tool call...")
    response_with_tool = """
    I need to search for information about machine learning.

    <tool_call>
    <tool_name>search_web</tool_name>
    <parameters>{"query": "machine learning algorithms"}</parameters>
    </tool_call>
    """
    
    result1 = adapter.parse_response_to_openai_format(response_with_tool)
    print(f"Result 1 success: {result1['success']}")

    # Test 2: Response without tool call
    print("\n3. Testing response without tool call...")
    response_without_tool = "I understand your question. Let me help you with that."
    
    result2 = adapter.parse_response_to_openai_format(response_without_tool)
    print(f"Result 2 success: {result2['success']}")

    # Test 3: Multiple tool calls
    print("\n4. Testing response with multiple tool calls...")
    response_multiple = """
    I need to search and calculate.

    <tool_call>
    <tool_name>search_web</tool_name>
    <parameters>{"query": "python programming"}</parameters>
    </tool_call>

    <tool_call>
    <tool_name>calculate</tool_name>
    <parameters>{"expression": "2 + 2"}</parameters>
    </tool_call>
    """
    
    result3 = adapter.parse_response_to_openai_format(response_multiple)
    print(f"Result 3 success: {result3['success']}")

    # Test 4: Invalid response
    print("\n5. Testing invalid response...")
    invalid_response = "This is just plain text without any tool calls."
    
    result4 = adapter.parse_response_to_openai_format(invalid_response)
    print(f"Result 4 success: {result4['success']}")

    # Test 5: Get OpenAI tools format
    print("\n6. Testing OpenAI tools format...")
    openai_tools = adapter.get_openai_tools_format()
    print(f"OpenAI tools count: {len(openai_tools)}")

    # Test 6: Convert to action
    print("\n7. Testing convert to action...")
    action = adapter.convert_to_action(response_with_tool)
    print(f"Action created: {action}")

    print("\n" + "=" * 60)
    print("Debug test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_debug_output()
