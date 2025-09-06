from __future__ import annotations

"""
String-based Radix Trie for efficient prefix matching and token caching.
Optimized for string prefixes with corresponding token IDs.
"""

import time
import threading
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Result of prefix matching operation."""

    matched_prefix: str
    token_ids: List[int]
    logp: List[float]
    remaining_string: str
    last_node: "StringTreeNode"


class StringTreeNode:
    """Tree node for string-based radix trie."""

    counter = 0

    def __init__(self, node_id: Optional[int] = None):
        # Core tree structure
        self.children: List[StringTreeNode] = []  # Use list to store children
        self.parent: Optional[StringTreeNode] = None

        # Node data
        self.string_key: str = ""  # The string fragment this node represents
        self.token_ids: Optional[List[int]] = None  # Token IDs for this node only (not cumulative)
        self.logp: Optional[List[float]] = None  # Log probabilities for this node's tokens

        # Access tracking
        self.last_access_time = time.monotonic()
        self.access_count = 0

        # Reference counting for protection from eviction
        self.ref_count = 0

        # Node identification
        self.id = StringTreeNode.counter if node_id is None else node_id
        StringTreeNode.counter += 1

    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf node."""
        return len(self.children) == 0

    @property
    def has_value(self) -> bool:
        """Check if this node has token IDs stored."""
        return self.token_ids is not None

    def validate_token_logp_consistency(self) -> bool:
        """Validate that token_ids and logp have consistent lengths."""
        if self.token_ids is None and self.logp is None:
            return True
        if self.token_ids is None or self.logp is None:
            return False
        return len(self.token_ids) == len(self.logp)

    @property
    def is_evictable(self) -> bool:
        """Check if this node can be evicted."""
        return self.ref_count == 0 and self.token_ids is not None

    def touch(self):
        """Update access time and count."""
        self.last_access_time = time.monotonic()
        self.access_count += 1

    def __lt__(self, other: StringTreeNode) -> bool:
        """For heap operations - least recently used first."""
        return self.last_access_time < other.last_access_time


class StringRadixTrie:
    """
    String-based Radix Trie for efficient prefix matching and token caching.

    Features:
    - Efficient string prefix matching
    - Token ID caching for matched prefixes
    - LRU-based eviction for memory management
    - Thread-safe operations
    - Automatic cleanup of stale entries
    """

    def __init__(
        self, max_cache_size: int = 10000, cleanup_interval: int = 300, enable_auto_cleanup: bool = True  # 5 minutes
    ):
        """
        Initialize the String Radix Trie.

        Args:
            max_cache_size: Maximum number of cached entries
            cleanup_interval: Interval in seconds for automatic cleanup
            enable_auto_cleanup: Whether to enable automatic cleanup
        """
        self.max_cache_size = max_cache_size
        self.cleanup_interval = cleanup_interval
        self.enable_auto_cleanup = enable_auto_cleanup

        # Tree structure
        self.root = StringTreeNode()
        self.root.string_key = ""
        self.root.ref_count = 1  # Root is always protected

        # Cache statistics
        self.total_entries = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Thread safety
        self._lock = threading.RLock()

        # Cleanup timer
        self._cleanup_timer = None
        if self.enable_auto_cleanup:
            self._start_cleanup_timer()

    def find_longest_prefix(self, text: str) -> MatchResult:
        """
        Find the longest cached prefix for the given text.

        Args:
            text: Input string to find prefix for

        Returns:
            MatchResult containing matched prefix, token IDs, logp, and remaining string
        """
        with self._lock:
            if not text:
                return MatchResult("", [], [], text, self.root)

            matched_tokens = []
            matched_logp = []
            matched_prefix = ""
            current_node = self.root
            remaining_text = text

            while remaining_text:
                # Find the best matching child that completely matches from start
                best_child = None
                best_key_len = 0

                for child_node in current_node.children:
                    # Only consider complete startswith matches using node's string_key
                    if remaining_text.startswith(child_node.string_key):
                        if len(child_node.string_key) > best_key_len:
                            best_child = child_node
                            best_key_len = len(child_node.string_key)

                if best_child is None:
                    # No complete startswith match found
                    break

                # Move to the best matching child
                best_child.touch()
                current_node = best_child
                matched_prefix += best_child.string_key
                remaining_text = remaining_text[best_key_len:]

                # Accumulate tokens and logp from this node
                if best_child.has_value:
                    matched_tokens.extend(best_child.token_ids)
                    matched_logp.extend(best_child.logp)
                    self.cache_hits += 1

            if not matched_tokens:
                self.cache_misses += 1

            return MatchResult(matched_prefix, matched_tokens, matched_logp, remaining_text, current_node)

    def insert(
        self,
        text: str,
        token_ids: List[int],
        logp: Optional[List[float]] = None,
        token_split_positions: Optional[List[int]] = None,
    ) -> bool:
        """
        Insert a string and its corresponding token IDs and log probabilities into the trie.

        Args:
            text: String to insert
            token_ids: Corresponding token IDs
            logp: Corresponding log probabilities (must match token_ids length)
            token_split_positions: Optional character positions where each token ends

        Returns:
            True if insertion was successful
        """
        with self._lock:
            if not text or not token_ids:
                return False

            # Validate logp consistency
            if logp is not None and len(logp) != len(token_ids):
                return False

            # Validate split positions if provided
            if token_split_positions is not None:
                if (
                    len(token_split_positions) != len(token_ids)
                    or token_split_positions[-1] > len(text)
                    or any(pos <= 0 for pos in token_split_positions)
                    or token_split_positions != sorted(token_split_positions)
                ):
                    return False

            # If logp is not provided, create default values (0.0)
            if logp is None:
                logp = [0.0] * len(token_ids)

            return self._insert_with_token_splits(text, token_ids, logp, token_split_positions)

    def _insert_with_token_splits(
        self, text: str, token_ids: List[int], logp: List[float], token_split_positions: Optional[List[int]] = None
    ) -> bool:
        """Insert with token splitting - cut tokens based on existing node lengths."""

        current_node = self.root
        remaining_text = text
        remaining_tokens = token_ids.copy()
        remaining_logp = logp.copy()

        while remaining_text:
            # Find best startswith match
            best_child = None
            best_key_len = 0

            for child_node in current_node.children:
                if remaining_text.startswith(child_node.string_key) and len(child_node.string_key) > best_key_len:
                    best_child = child_node
                    best_key_len = len(child_node.string_key)

            if best_child is not None:
                assert best_child.has_value, "Node must have tokens"
                # Node already has tokens, cut from remaining based on its length
                node_token_len = len(best_child.token_ids)
                if node_token_len <= len(remaining_tokens):
                    remaining_tokens = remaining_tokens[node_token_len:]
                    remaining_logp = remaining_logp[node_token_len:]

                current_node = best_child
                remaining_text = remaining_text[best_key_len:]
            else:
                # Create new node with all remaining tokens
                new_node = StringTreeNode()
                new_node.parent = current_node
                new_node.string_key = remaining_text

                # Give all remaining tokens to the new node
                if remaining_tokens:
                    new_node.token_ids = remaining_tokens
                    new_node.logp = remaining_logp
                    new_node.touch()

                current_node.children.append(new_node)  # Add to children list
                self.total_entries += 1
                break

        return True

    def remove(self, text: str) -> bool:
        """
        Remove a string and all nodes with this text as prefix from the trie.

        Args:
            text: String to remove (will also remove all strings starting with this text)

        Returns:
            True if any removal was performed
        """
        with self._lock:
            node = self._find_node_by_text(text)
            if node:
                removed_count = self._clean_node_subtree(node)
                return removed_count > 0
            return False

    def _find_node_by_text(self, text: str) -> Optional[StringTreeNode]:
        """
        Find node by exact text match.

        Args:
            text: Text to find

        Returns:
            Node if found, None otherwise
        """
        result = self.find_longest_prefix(text)
        if result.matched_prefix == text:
            return result.last_node
        return None

    def cleanup_stale_entries(self, max_age_seconds: int = 3600) -> int:
        """Clean up stale entries."""
        with self._lock:
            stale_nodes = self._find_stale_nodes(max_age_seconds)
            removed_count = 0
            for node in stale_nodes:
                removed_count += self._clean_node_subtree(node)
            return removed_count

    def _find_stale_nodes(self, max_age_seconds: int) -> List[StringTreeNode]:
        """
        Find all stale nodes by checking layer by layer.
        If a parent node is stale, children are not checked.

        Args:
            max_age_seconds: Maximum age before considering stale

        Returns:
            List of stale nodes to remove
        """
        current_time = time.monotonic()
        stale_nodes = []

        def check_node(node):
            if node == self.root:
                # Check children but root itself is never stale
                for child in node.children:
                    check_node(child)
                return

            # If this node is stale, add it and skip children
            if node.has_value and current_time - node.last_access_time > max_age_seconds and node.is_evictable:
                stale_nodes.append(node)
                return  # Don't check children

            # Node is not stale, check its children
            for child in node.children:
                check_node(child)

        check_node(self.root)
        return stale_nodes

    def _clean_node_subtree(self, node: StringTreeNode) -> int:
        """
        Clean a node and all its descendants.
        This is the core cleanup function.

        Args:
            node: Node to clean (including all descendants)

        Returns:
            Number of nodes removed
        """
        if node == self.root:
            return 0
        return self._remove_node_and_descendants(node)

    def _remove_node_and_descendants(self, node: StringTreeNode) -> int:
        """
        Remove a node and all its descendants from the trie.

        Args:
            node: The node to remove along with all its descendants

        Returns:
            Number of nodes removed
        """
        if node == self.root:
            # Never remove root node
            return 0

        removed_count = 0

        # First, recursively remove all descendants
        for child in list(node.children):  # Create a copy to avoid modification during iteration
            removed_count += self._remove_node_and_descendants(child)

        # Count this node if it has data
        if node.has_value:
            removed_count += 1

        # Remove this node from its parent
        if self._remove_node_from_parent(node):
            # Update count for the node structure itself
            pass  # _remove_node_from_parent already decrements total_entries

        return removed_count

    def _remove_node_from_parent(self, node: StringTreeNode) -> bool:
        """Remove a node from its parent's children list."""
        if node.parent and node in node.parent.children:
            node.parent.children.remove(node)
            self.total_entries -= 1
            return True
        return False

    def _start_cleanup_timer(self):
        """Start the automatic cleanup timer."""
        self.cleanup_stale_entries()
        if self.enable_auto_cleanup:
            self._cleanup_timer = threading.Timer(self.cleanup_interval, self._start_cleanup_timer)
            self._cleanup_timer.daemon = True
            self._cleanup_timer.start()

    def stop_cleanup_timer(self):
        """Stop the automatic cleanup timer."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

            return {
                "total_entries": self.total_entries,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": hit_rate,
                "max_cache_size": self.max_cache_size,
            }

    def clear(self):
        """Clear all entries from the trie."""
        with self._lock:
            self.root = StringTreeNode()
            self.root.string_key = ""
            self.root.ref_count = 1
            self.total_entries = 0
            self.cache_hits = 0
            self.cache_misses = 0

    def pretty_print(self):
        """Print the trie structure in a readable format."""
        print("String Radix Trie Structure:")
        print("=" * 50)
        self._print_node(self.root, 0)
        print("=" * 50)
        stats = self.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

    def _print_node(self, node: StringTreeNode, depth: int):
        """Recursively print node structure."""
        indent = "  " * depth
        key_repr = repr(node.string_key) if node.string_key else "<root>"
        token_info = ""
        if node.has_value:
            token_info = f" -> tokens: {node.token_ids}"
            if node.logp:
                token_info += f", logp: {[round(p, 3) for p in node.logp]}"
        access_info = f" (accessed: {node.access_count}, ref: {node.ref_count})"

        print(f"{indent}{key_repr}{token_info}{access_info}")

        for child in node.children:
            self._print_node(child, depth + 1)

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_cleanup_timer()


# Example usage and testing
if __name__ == "__main__":
    # Create trie instance
    trie = StringRadixTrie(max_cache_size=100)

    # Example usage with simplified insert
    test_cases = [
        ("Hello world", [1, 2, 3], [-0.1, -0.2, -0.3]),
        ("Hello", [1, 2], [-0.1, -0.2]),
        ("Hi there", [4, 5, 6], [-0.4, -0.5, -0.6]),
    ]

    # Insert test data
    print("Inserting test data...")
    for text, tokens, logp in test_cases:
        success = trie.insert(text, tokens, logp)
        print(f"Inserted '{text}' -> {tokens}: {success}")

    print("\nTrie structure:")
    trie.pretty_print()

    # Test prefix matching
    print("\nTesting prefix matching:")
    test_queries = [
        "Hello world!",  # Should match "Hello world" completely
        "Hello everyone",  # Should match "Hello" only
        "Hi there",  # Should match "Hi" only
        "How are you doing?",  # Should match "How are you" completely
        "Goodbye",  # Should not match anything
        "Hell",  # Should not match anything (not complete startswith)
    ]

    for query in test_queries:
        result = trie.find_longest_prefix(query)
        print(f"Query: '{query}'")
        print(f"  Matched: '{result.matched_prefix}' -> tokens: {result.token_ids}, logp: {result.logp}")
        print(f"  Remaining: '{result.remaining_string}'")
        print()

    # Test removal
    print("Testing removal:")
    removed = trie.remove("Hello")
    print(f"Removed 'Hello': {removed}")

    result = trie.find_longest_prefix("Hello world")
    print(
        f"After removal - 'Hello world' -> matched: '{result.matched_prefix}', tokens: {result.token_ids}, logp: {result.logp}"
    )

    # Show final stats
    print("\nFinal statistics:")
    stats = trie.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Cleanup
    trie.stop_cleanup_timer()
