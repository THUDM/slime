from __future__ import annotations

"""
String-based Radix Trie for efficient prefix matching and token caching.
Optimized for string prefixes with corresponding token IDs.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MatchResult:
    """Result of prefix matching operation."""

    matched_prefix: str
    token_ids: List[int]
    logp: List[float]
    loss_mask: List[int]  # Added loss mask for model generation parts
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
        self.loss_mask: Optional[List[int]] = None  # Loss mask for model generation parts

        # Access tracking
        self.last_access_time = time.monotonic()
        self.access_count = 0

        # Reference counting for protection from eviction
        self.ref_count = 0

        # Weight version tracking
        self.weight_version: Optional[int] = None  # Weight version for this node

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
        """Validate that token_ids, logp, and loss_mask have consistent lengths."""
        if self.token_ids is None and self.logp is None and self.loss_mask is None:
            return True

        # Check if at least one is not None
        if self.token_ids is not None and len(self.token_ids) > 0:
            token_len = len(self.token_ids)
            if self.logp is not None and len(self.logp) != token_len:
                return False
            if self.loss_mask is not None and len(self.loss_mask) != token_len:
                return False

        return True

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
    - Thread-safe operations
    - Weight version tracking
    - Automatic garbage collection based on weight version thresholds
    """

    def __init__(self, max_cache_size: int = 10000, gc_threshold_k: int = 5, tokenizer=None, verbose: bool = False):
        """
        Initialize the String Radix Trie.
        Args:
            max_cache_size: Maximum number of cached token IDs (triggers GC when exceeded)
            gc_threshold_k: GC threshold - nodes with weight_version < (current_version - k) will be removed
            tokenizer: Optional tokenizer for converting text to tokens when not found in cache
            verbose: Whether to print debug information and tree structure
        """
        self.max_cache_size = max_cache_size
        self.gc_threshold_k = gc_threshold_k
        self.tokenizer = tokenizer
        self.verbose = verbose

        # Tree structure
        self.root = StringTreeNode()
        self.root.string_key = ""
        self.root.ref_count = 1  # Root is always protected

        # Cache statistics
        self.total_entries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cur_cache_size = 0  # Total number of token IDs across all nodes

        # Thread safety
        self._lock = threading.RLock()

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
                return MatchResult("", [], [], [], text, self.root)

            matched_tokens = []
            matched_logp = []
            matched_loss_mask = []
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

                # Accumulate tokens, logp, and loss_mask from this node
                if best_child.has_value:
                    matched_tokens.extend(best_child.token_ids)
                    matched_logp.extend(best_child.logp)
                    if best_child.loss_mask is not None:
                        matched_loss_mask.extend(best_child.loss_mask)
                    else:
                        # If no loss_mask is stored, create default mask same as logp
                        matched_loss_mask.extend([1] * len(best_child.token_ids))
                    self.cache_hits += 1

            if not matched_tokens:
                self.cache_misses += 1

            result = MatchResult(
                matched_prefix, matched_tokens, matched_logp, matched_loss_mask, remaining_text, current_node
            )

            # Print tree structure if verbose is enabled
            if self.verbose:
                print("Tree structure after find_longest_prefix:")
                self.pretty_print()

            return result

    def insert(
        self,
        text: str,
        token_ids: List[int],
        logp: Optional[List[float]] = None,
        loss_mask: Optional[List[int]] = None,
        weight_version: Optional[int] = None,
    ) -> bool:
        """
        Insert a string and its corresponding token IDs, log probabilities, and loss mask into the trie.
        Args:
            text: String to insert
            token_ids: Corresponding token IDs
            logp: Corresponding log probabilities (must match token_ids length)
            loss_mask: Corresponding loss mask for model generation parts (must match token_ids length)
            weight_version: Optional weight version for this insertion
        Returns:
            True if insertion was successful
        """
        with self._lock:
            if not text or not token_ids:
                if self.verbose:
                    print("[RadixTree] Insertion failed: text or token_ids is empty")
                return False

            # Use provided weight version
            current_weight_version = weight_version

            # Validate logp consistency
            if logp is not None and len(logp) != len(token_ids):
                if self.verbose:
                    print(
                        f"[WARNING] Logp length {len(logp)} does not match token length {len(token_ids)} for text: {text}"
                    )
                    print(f"[WARNING] Logp: {logp}")
                    print(f"[WARNING] Token IDs: {token_ids}")
                return False

            # Validate loss_mask consistency
            if loss_mask is not None and len(loss_mask) != len(token_ids):
                if self.verbose:
                    print(
                        f"[WARNING] Loss mask length {len(loss_mask)} does not match token length {len(token_ids)} for text: {text}"
                    )
                    print(f"[WARNING] Loss mask: {loss_mask}")
                    print(f"[WARNING] Token IDs: {token_ids}")
                return False

            # If logp is not provided, create default values (0.0)
            if logp is None:
                logp = [0.0] * len(token_ids)

            # If loss_mask is not provided, create default values (1 for model generation parts)
            if loss_mask is None:
                loss_mask = [0] * len(token_ids)

            result = self._insert(text, token_ids, logp, loss_mask, current_weight_version)

            # Check if GC should be triggered after insert
            if self.cur_cache_size > self.max_cache_size and weight_version is not None:
                if self.verbose:
                    print(
                        f"[RadixTree] Cache size {self.cur_cache_size} exceeds limit {self.max_cache_size}, triggering GC"
                    )
                gc_removed = self.gc_by_weight_version(weight_version)
                if self.verbose:
                    print(f"[RadixTree] GC removed {gc_removed} nodes, new cache size: {self.cur_cache_size}")

            # Print tree structure if verbose is enabled
            if self.verbose:
                print("Tree structure after insert:")
                self.pretty_print()

            return result

    def _insert(
        self,
        text: str,
        token_ids: List[int],
        logp: List[float],
        loss_mask: List[int],
        weight_version: Optional[int] = None,
    ) -> bool:
        """Insert tokens - skip tokens for existing nodes just like we skip text."""

        current_node = self.root
        remaining_text = text
        remaining_tokens = token_ids[:]  # Copy the tokens list
        remaining_logp = logp[:]  # Copy the logp list
        remaining_loss_mask = loss_mask[:]  # Copy the loss_mask list

        # Track all nodes traversed during insert for weight version update
        traversed_nodes = [current_node]
        new_node = None

        while remaining_text:
            # Find best startswith match
            best_child = None
            best_key_len = 0

            for child_node in current_node.children:
                if remaining_text.startswith(child_node.string_key) and len(child_node.string_key) > best_key_len:
                    best_child = child_node
                    best_key_len = len(child_node.string_key)

            if best_child is not None:
                # Found existing node - skip its text and tokens
                current_node = best_child
                traversed_nodes.append(current_node)
                remaining_text = remaining_text[best_key_len:]

                # Skip the tokens that this existing node covers
                if best_child.has_value:
                    tokens_to_skip = len(best_child.token_ids)
                    remaining_tokens = remaining_tokens[tokens_to_skip:]
                    remaining_logp = remaining_logp[tokens_to_skip:]
                    remaining_loss_mask = remaining_loss_mask[tokens_to_skip:]
            else:
                # Create new node for remaining text with remaining tokens
                new_node = StringTreeNode()
                new_node.parent = current_node
                new_node.string_key = remaining_text

                if remaining_tokens:  # Only assign if there are tokens left
                    new_node.token_ids = remaining_tokens
                    new_node.logp = remaining_logp
                    new_node.loss_mask = remaining_loss_mask
                    new_node.touch()
                    # Increment cache size by number of tokens added
                    self.cur_cache_size += len(remaining_tokens)

                current_node.children.append(new_node)
                traversed_nodes.append(new_node)
                self.total_entries += 1
                break

        # If we've traversed the entire text and the last node doesn't have tokens,
        # assign remaining tokens to it
        if remaining_text == "" and not current_node.has_value:
            if remaining_tokens:  # Only assign if there are tokens left
                current_node.token_ids = remaining_tokens
                current_node.logp = remaining_logp
                current_node.loss_mask = remaining_loss_mask
                current_node.touch()
                self.cur_cache_size += len(remaining_tokens)

        # Update weight version for all traversed nodes
        if weight_version is not None and new_node:
            new_node.weight_version = weight_version

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

                # Print tree structure if verbose is enabled
                if self.verbose:
                    print("Tree structure after remove:")
                    self.pretty_print()

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

        # Count this node if it has data and decrement cache size
        if node.has_value:
            removed_count += 1
            # Decrement cache size by number of tokens removed
            self.cur_cache_size -= len(node.token_ids)

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

    def gc_by_weight_version(self, current_weight_version: Optional[int] = None) -> int:
        """
        Perform garbage collection based on weight version.
        Remove nodes with weight_version < (current_weight_version - gc_threshold_k).
        Args:
            current_weight_version: Current weight version to use for GC threshold
        Returns:
            Number of nodes removed
        """
        with self._lock:
            if current_weight_version is None:
                if self.verbose:
                    print("[RadixTree GC] No weight version provided, skipping GC")
                return 0

            gc_threshold = current_weight_version - self.gc_threshold_k
            if self.verbose:
                print(
                    f"[RadixTree GC] Starting GC with threshold: {gc_threshold} (current_version: {current_weight_version}, k: {self.gc_threshold_k})"
                )

            nodes_to_remove = self._find_outdated_nodes(gc_threshold)
            removed_count = 0

            for node in nodes_to_remove:
                # Validate that subtree weight versions are <= parent weight version
                self._validate_subtree_weight_versions(node)
                removed_count += self._clean_node_subtree(node)

            if self.verbose:
                print(f"[RadixTree GC] Completed GC, removed {removed_count} nodes")

            return removed_count

    def _find_outdated_nodes(self, gc_threshold: int) -> List[StringTreeNode]:
        """
        Find nodes that should be removed based on weight version threshold.
        Uses layer-by-layer traversal - if parent is outdated, children are not checked.
        Args:
            gc_threshold: Weight version threshold (nodes < this value will be removed)
        Returns:
            List of nodes to remove
        """
        outdated_nodes = []

        def check_node(node):
            if node == self.root:
                # Root is never removed, check its children
                for child in node.children:
                    check_node(child)
                return

            # Check if this node should be removed
            if node.weight_version is not None and node.weight_version <= gc_threshold and node.has_value:
                outdated_nodes.append(node)
                return  # Don't check children since entire subtree will be removed

            # Node is not outdated, check its children
            for child in node.children:
                check_node(child)

        check_node(self.root)
        return outdated_nodes

    def _validate_subtree_weight_versions(self, node: StringTreeNode):
        """
        Validate that all nodes in subtree have weight_version <= parent weight_version.
        Args:
            node: Root node of subtree to validate
        """

        def validate_recursive(current_node, parent_weight_version):
            if current_node.weight_version is not None and parent_weight_version is not None:
                assert current_node.weight_version <= parent_weight_version, (
                    f"Child node weight_version {current_node.weight_version} > "
                    f"parent weight_version {parent_weight_version}"
                )

            # Recursively validate children
            for child in current_node.children:
                validate_recursive(child, current_node.weight_version)

        # Start validation from the node itself
        validate_recursive(node, node.weight_version)

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
                "cur_cache_size": self.cur_cache_size,
                "gc_threshold_k": self.gc_threshold_k,
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
            self.cur_cache_size = 0

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
            if node.loss_mask:
                token_info += f", loss_mask: {node.loss_mask}"
        access_info = f" (accessed: {node.access_count}, ref: {node.ref_count})"

        print(f"{indent}{key_repr}{token_info}{access_info}")

        for child in node.children:
            self._print_node(child, depth + 1)

    def retrieve_from_text(self, text: str, return_logprob: bool = True):
        """
        Get tokens from text by looking up in radix tree or using tokenizer.
        Also fetches weight version from worker during this operation.
        Args:
            text: Input text to get tokens for
            return_logprob: If True, also return log probabilities
        Returns:
            List of token IDs corresponding to the input text if return_logprob is False.
            Tuple of (token_ids, logp) if return_logprob is True.
        """
        # Call find_longest_prefix to get the match result
        result = self.find_longest_prefix(text)

        # If we have a match and it covers the entire text, return the tokens
        if result.matched_prefix and result.token_ids:
            additional_tokens = self.tokenizer(result.remaining_string, add_special_tokens=False)["input_ids"]
            return (
                result.token_ids + additional_tokens,
                (
                    result.logp + len(additional_tokens) * [0.0]
                    if return_logprob
                    else [0] * len(result.token_ids + additional_tokens)
                ),
                result.loss_mask + len(additional_tokens) * [0],
            )
        # If result is empty and input text is not empty, tokenize with tokenizer
        # This is needed because we cannot get the prompt token id from engine response
        # We have to manually insert the text and token into the tree
        if self.tokenizer and text:
            # Tokenize the text using the provided tokenizer
            tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            # Insert the text and tokens into the tree
            self.insert(text, tokens)
            # Return the tokens
            return (tokens, [0.0] * len(tokens), [0] * len(tokens))
        else:
            raise ValueError("Tokenizer or input text can't be empty")


# Example usage and testing
if __name__ == "__main__":
    # Create trie instance for testing
    trie = StringRadixTrie(max_cache_size=100, verbose=True)

    # Test token retrieval
    print("\nTesting token retrieval:")
    test_tokens = trie.retrieve_from_text("Hello world")
    print(f"Tokens for 'Hello world': {test_tokens}")

    # Example usage with simplified insert
    test_cases = [
        ("Hello world", [1, 2, 3], [-0.1, -0.2, -0.3]),
        ("Hello", [1, 2], [-0.1, -0.2]),
        ("Hi there", [4, 5, 6], [-0.4, -0.5, -0.6]),
    ]

    # Insert test data with weight version and loss masks
    print("Inserting test data...")
    for text, tokens, logp in test_cases:
        # Create loss_mask to match tokens length, 1 for model generation parts
        loss_mask = [1] * len(tokens)
        success = trie.insert(text, tokens, logp, loss_mask, weight_version=1)
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
        print(
            f"  Matched: '{result.matched_prefix}' -> tokens: {result.token_ids}, logp: {result.logp}, loss_mask: {result.loss_mask}"
        )
        print(f"  Remaining: '{result.remaining_string}'")
        print()

    # Test removal
    print("Testing removal:")
    removed = trie.remove("Hello")
    print(f"Removed 'Hello': {removed}")

    result = trie.find_longest_prefix("Hello world")
    print(
        f"After removal - 'Hello world' -> matched: '{result.matched_prefix}', tokens: {result.token_ids}, logp: {result.logp}, loss_mask: {result.loss_mask}"
    )

    # Show final stats
    print("\nFinal statistics:")
    stats = trie.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Test GC with weight version
    print("\nTesting GC with weight version 5:")
    gc_removed = trie.gc_by_weight_version(5)
    print(f"GC removed {gc_removed} nodes")
