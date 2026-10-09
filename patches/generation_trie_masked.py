# Native repeat-masked decoding support for GRAM.
#
# CountedTrie augments the plain generation trie with per-node sequence counts
# so that a *blocked* subset (the items already present in a user's history
# window) can be subtracted at constraint time without rebuilding the global
# candidate trie for every user.
#
# A token t may follow prefix p iff
#     count_global(p + [t]) > count_blocked(p + [t])
# i.e. at least one non-blocked candidate sequence passes through that node.
# Native repeat-masked decoding support for GRAM.
from __future__ import annotations

from typing import Dict, List, Sequence


class CountedTrieNode:
    __slots__ = ("children", "count")

    def __init__(self):
        self.children: Dict[int, "CountedTrieNode"] = {}
        self.count: int = 0


class CountedTrie:
    def __init__(self, sequences: Sequence[Sequence[int]] = ()):  # noqa: D401
        self.root = CountedTrieNode()
        for sequence in sequences:
            self.add(sequence)

    def add(self, sequence: Sequence[int]) -> None:
        node = self.root
        node.count += 1
        for token in sequence:
            node = node.children.setdefault(token, CountedTrieNode())
            node.count += 1

    def _walk(self, prefix: Sequence[int]):
        node = self.root
        for token in prefix:
            node = node.children.get(token)
            if node is None:
                return None
        return node

    def allowed_after(self, prefix: Sequence[int], blocked: "CountedTrie | None"):
        node = self._walk(prefix)
        if node is None:
            return []
        blocked_node = blocked._walk(prefix) if blocked is not None else None
        out = []
        for token, child in node.children.items():
            blocked_count = 0
            if blocked_node is not None:
                blocked_child = blocked_node.children.get(token)
                if blocked_child is not None:
                    blocked_count = blocked_child.count
            if child.count > blocked_count:
                out.append(token)
        return out


def masked_prefix_allowed_tokens_fn(global_trie: CountedTrie, blocked_trie: CountedTrie | None):
    def prefix_allowed_tokens(batch_id, sentence):
        return global_trie.allowed_after(sentence.tolist(), blocked_trie)

    return prefix_allowed_tokens
