#!/usr/bin/env python3

import collections

# Trie representation of a vocabulary.
class Trie:
    def __init__(self, value=None, enforce_token_maximality=True):
        self._children = collections.defaultdict(
                lambda: Trie(enforce_token_maximality=enforce_token_maximality))
        self._value = value
        self._enforce_token_maximality = enforce_token_maximality

    def insert(self, key, value, depth=0):
        if len(key) == depth:
            self._value = value
        else:
            self._children[key[depth]].insert(key, value, depth + 1)

    @staticmethod
    def from_vocabulary(vocab: list[str], enforce_token_maximality: bool = True):
        t = Trie(enforce_token_maximality=enforce_token_maximality)

        for i, token in enumerate(vocab):
            if token:
                t.insert(token, i)

        return t

    def antimonotonic_filter(self, predicate, key='') -> list[str]:
        this_node_valid = predicate(key)

        if not this_node_valid:
            # Prune using anti-monotonicity: no children will be valid.
            return []

        children_values = []

        for k, c in self._children.items():
            children_values.extend(c.antimonotonic_filter(predicate, key + k))

        this_value = [self._value] if self._value is not None else []

        if self._enforce_token_maximality:
            # Only return maximal strings.
            if len(children_values) or self._value is None:
                return children_values
            return this_value

        return this_value + children_values
