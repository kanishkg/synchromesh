#!/usr/bin/env python3

import regex

from completion_engine import CompletionEngine, LarkCompletionEngine
from language_model import LanguageModel, RandomLanguageModel


# Implements the Constrained Semantic Decoding algorithm.
def predict_constrained(completion_engine: CompletionEngine, lm: LanguageModel) -> str:
    completion_points: dict[str, regex.Pattern] = {}

    completion_points[''] = completion_engine.complete('')

    prediction = ''

    while not completion_engine.is_complete(prediction):
        valid_tokens = []

        for i, token in enumerate(lm.vocabulary()):
            if is_prefix_valid(completion_engine, completion_points, prediction + token):
                valid_tokens.append(i)

        assert len(valid_tokens) > 0, f"No valid tokens for {repr(prediction)}"
        predicted_token = lm.predict_token(prediction, valid_tokens)
        prediction += lm.vocabulary()[predicted_token]
    return prediction


def is_prefix_valid(completion_engine: CompletionEngine,
                    completion_points: dict[str, regex.Pattern],
                    s: str) -> bool:

    # 1- Find longest completion point that is a prefix of s.
    longest_completion_point = 0

    for i in range(len(s)):
        if s[:i] in completion_points:
            longest_completion_point = i

    # 2- Take the 'remainder'.
    completion_point_regex = completion_points[s[:longest_completion_point]]
    remainder = s[longest_completion_point:]

    # 3- Feed it character by character to the regex given by the completion point, and handle 3 cases:
    for i in range(len(remainder)):
        # If we have a violation of the regex.
        if not completion_point_regex.fullmatch(remainder[:i+1], partial=True):
            # Check if we have a full match up to the previous character.
            if completion_point_regex.fullmatch(remainder[:i]):
                # We found another completion point, reduce the problem and call recursively.
                new_completion_point = s[:longest_completion_point] + remainder[:i]
                new_completion_point_regex = completion_engine.complete(new_completion_point)
                completion_points[new_completion_point] = new_completion_point_regex
                return is_prefix_valid(completion_engine, completion_points, s)
            else:
                return False

    #    Case c- Got to the end with no violations, return True
    return True

if __name__ == "__main__":
    json_grammar = r"""
        ?value: dict
            | list
            | string
            | SIGNED_NUMBER      -> number
            | "true"             -> true
            | "false"            -> false
            | "null"             -> null

        list : "[" [value ("," value)*] "]"

        dict : "{" [pair ("," pair)*] "}"
        pair : string ":" value

        string : "\"" /[A-Z]{3}/ "\""

        %import common.ESCAPED_STRING
        %import common.SIGNED_NUMBER
        %import common.WS
        %ignore WS

        """
    for i in range(1000):
        json_comp_engine = LarkCompletionEngine(json_grammar, 'value')
        rlm = RandomLanguageModel()
        print(predict_constrained(json_comp_engine, rlm))