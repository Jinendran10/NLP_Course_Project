"""Greedy longest-match phrase lookup over token sequences, backed by a token trie."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Generic, TypeVar

V = TypeVar("V")


class PhraseMatcher(Generic[V]):
    """Finds non-overlapping occurrences of known phrases in a token sequence.

    Scanning is left to right; at each position the longest phrase wins, so
    "React Native" is matched as one skill rather than "React" + "Native".
    """

    def __init__(self, phrases: Mapping[tuple[str, ...], V]) -> None:
        # Nested dicts keyed by token; the value of a complete phrase is stored
        # under the key None (tokens are never None).
        self._root: dict[Any, Any] = {}
        for phrase, value in phrases.items():
            if not phrase:
                raise ValueError("cannot index an empty phrase")
            node = self._root
            for token in phrase:
                node = node.setdefault(token, {})
            node[None] = value

    def find(self, tokens: Sequence[str]) -> list[tuple[int, int, V]]:
        """Return (start, end, value) for each match; end is exclusive."""
        matches: list[tuple[int, int, V]] = []
        i, n = 0, len(tokens)
        while i < n:
            node, best, j = self._root, None, i
            while j < n and (node := node.get(tokens[j])) is not None:
                j += 1
                if None in node:
                    best = (i, j, node[None])
            if best is None:
                i += 1
            else:
                matches.append(best)
                i = best[1]
        return matches
