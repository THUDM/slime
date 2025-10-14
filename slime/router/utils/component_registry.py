"""
Component Registry for Slime Router.

This module provides a centralized registry for managing shared components
like tokenizer and radix tree instances across different modules.

Design Principles:
- Zero fallback: Components must be explicitly registered
- Fast failure: Clear error messages when components are missing
- Simple API: Easy to use and test
- Thread safety: Safe for concurrent access from multiple threads
"""

import threading
from typing import Any, Dict, Optional, Callable
from contextlib import contextmanager


class ComponentRegistry:
    """
    Thread-safe component registry for managing shared instances.

    Provides dependency injection capabilities without magic strings
    or complex fallback logic. Components must be explicitly registered
    before they can be retrieved. All operations are thread-safe.
    """

    def __init__(self):
        """Initialize the thread-safe component registry."""
        self._components: Dict[str, Any] = {}
        self._destructors: Dict[str, Callable] = {}
        self._lock = threading.RLock()  # Reentrant lock for recursive access

    def register(self, name: str, instance: Any, on_destroy: Optional[Callable] = None) -> None:
        """
        Thread-safe component registration with optional destructor.

        Args:
            name: Component name for later retrieval
            instance: Component instance to register
            on_destroy: Optional destructor function called when component is removed

        Raises:
            ValueError: If name is empty or instance is None
            RuntimeError: If component with same name already exists
        """
        with self._lock:
            if not name or not name.strip():
                raise ValueError("Component name cannot be empty")

            if instance is None:
                raise ValueError("Component instance cannot be None")

            # Protect against overwriting existing components
            if name in self._components:
                raise RuntimeError(
                    f"Component '{name}' already registered. "
                    f"Use remove() first if you want to replace it."
                )

            self._components[name] = instance
            if on_destroy:
                self._destructors[name] = on_destroy

    def get(self, name: str, default: Any = None) -> Any:
        """
        Thread-safe component retrieval with optional default value.

        Args:
            name: Name of the component to retrieve
            default: Default value to return if component is not found (default: None)

        Returns:
            The registered component instance, or default value if not found

        Raises:
            RuntimeError: If component is not found and no default provided
            ValueError: If name is empty
        """
        with self._lock:
            if not name or not name.strip():
                raise ValueError("Component name cannot be empty")

            if name not in self._components:
                # Return default if provided
                if default is not None:
                    return default

                # Only raise if no default provided
                available_components = list(self._components.keys())
                raise RuntimeError(
                    f"Required component '{name}' not found. "
                    f"Available components: {available_components}"
                )

            return self._components[name]

    def has(self, name: str) -> bool:
        """
        Thread-safe component existence check.

        Args:
            name: Name of the component to check

        Returns:
            True if component is registered, False otherwise
        """
        with self._lock:
            return name in self._components

    def remove(self, name: str) -> bool:
        """
        Thread-safe component removal with destructor call.

        Args:
            name: Name of the component to remove

        Returns:
            True if component was removed, False if it didn't exist
        """
        with self._lock:
            if name not in self._components:
                return False

            self._destroy_component(name)
            del self._components[name]
            return True

    def list_components(self) -> list[str]:
        """
        Get list of all registered component names (thread-safe snapshot).

        Returns:
            List of component names
        """
        with self._lock:
            return list(self._components.keys())

    def list(self) -> Dict[str, Any]:
        """
        Get dictionary of all registered components (thread-safe snapshot).

        Returns:
            Dictionary mapping component names to component instances
        """
        with self._lock:
            return dict(self._components)

    def clear(self) -> None:
        """Clear all registered components with destructors. Mainly for testing."""
        with self._lock:
            for name in list(self._components.keys()):
                self._destroy_component(name)
            self._components.clear()
            self._destructors.clear()

    def _destroy_component(self, name: str) -> None:
        """
        Safely destroy a component by calling its destructor if available.

        This method should only be called while holding the lock.
        """
        if name in self._destructors:
            try:
                self._destructors[name](self._components[name])
            except Exception as e:
                # Log error but don't raise to avoid breaking cleanup
                print(f"Warning: Error destroying component '{name}': {e}")
            finally:
                del self._destructors[name]

    @contextmanager
    def transaction(self):
        """
        Provide transactional access to multiple components.

        Usage:
            with registry.transaction() as reg:
                component1 = reg.get("comp1")
                component2 = reg.get("comp2")
                # Both operations are performed under the same lock
        """
        with self._lock:
            yield self

    def get_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics about registered components.

        Returns:
            Dictionary with component statistics
        """
        with self._lock:
            return {
                "total_components": len(self._components),
                "components_with_destructors": len(self._destructors),
                "component_names": list(self._components.keys())
            }