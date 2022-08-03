#!/usr/bin/env python3

import regex

from completion_engine import CompletionEngine
from language_model import LanguageModel


# Implements the Constrained Semantic Decoding algorithm.
def predict_constrained(completion_engine: CompletionEngine, lm: LanguageModel) -> str:
    completion_points: dict[str, regex.Pattern] = {}

    completion_points[''] = completion_engine.complete('')

    done = False
    prediction = ''

    while not done:
        valid_tokens = []

        for i, token in enumerate(lm.vocabulary()):
            if is_prefix_valid(completion_engine, completion_points, prediction + token):
                valid_tokens.append(i)

        predicted_token = lm.predict_token(prediction, valid_tokens)

        if predicted_token == lm.stop_token():
            done = True
        else:
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
