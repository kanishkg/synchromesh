#!/usr/bin/env python3

import os
import regex
import time

from completion_engine import CompletionEngine, LarkCompletionEngine
from language_model import LanguageModel, RandomLanguageModel, OpenAIModel


# Implements the Constrained Semantic Decoding algorithm.
def predict_constrained(completion_engine: CompletionEngine, lm: LanguageModel,
                        top_k: int = 1, verbose: bool = True,
                        batch_size: int = 50, stop_tokens: list[str]=None) -> str:
    completion_points: dict[str, regex.Pattern] = {}

    completion_points[''] = completion_engine.complete('')

    prediction = ''

    while not completion_engine.is_complete(prediction):
        # Ask for unconstrained prediction.
        continuation = lm.predict_unconstrained(prediction, batch_size, stop=stop_tokens)
        # hacky way to filter newlines
        continuation = continuation.replace('\n', '')
        found_violation = False
        if verbose:
            print(f"continuation: {repr(continuation)}")
        for token in lm.tokenize(continuation):
            if is_prefix_valid(completion_engine, completion_points,
                               prediction + lm.get_token(token)):
                prediction += lm.get_token(token)
            else:
                if completion_engine.is_complete(prediction):
                    break
                found_violation = True
                if verbose:
                    print(f"found violation at token: {lm.get_token(token)}")
                    print(f"valid prefix: {prediction}")
                break

        if found_violation:
            # Do constrained prediction for next token.
            valid_tokens = []

            if verbose:
                print(f"constrained prediction for: {prediction}")
            for i, t in enumerate(lm.vocabulary()):
                if is_prefix_valid(completion_engine, completion_points, prediction + t):
                    valid_tokens.append(i)

            assert len(valid_tokens) > 0, f"No valid tokens after {repr(prediction)}"
            predictions, probabilities = lm.predict_token(prediction, valid_tokens, top_k)

            if verbose:
                print(f"current prediction: {prediction}")
                print(f"Top {min(top_k, len(valid_tokens))} next tokens:")
                for i, (t_idx, prob) in enumerate(zip(predictions, probabilities)):
                    print(f'{i+1}. {lm.get_token(t_idx)} {prob}')

            predicted_token = predictions[0]
            prediction += lm.get_token(predicted_token)
    return prediction


def is_prefix_valid(completion_engine: CompletionEngine,
                    completion_points: dict[str, regex.Pattern],
                    s: str) -> bool:

    # 1- Find longest completion point that is a prefix of s.
    longest_completion_point = 0

    for i in range(len(s)+1):
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
                # check if this base case is correct. (this is the case for the terminal symbol)
                if completion_engine.is_complete(new_completion_point):
                    return False
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

    college_grammar = r"""
        ?request: function "of" dept code
        function: "instructor" | "students" | "capacity" | "department" | "school" | "college"
        dept:  /[A-Z]{3}/
        code: /[0-9]{3}/
        %import common.WS
        %ignore WS
    """

    college_prompt = """Paraphrase the following sentences
Human:who teaches CSE101?
Bot:instructor of CSE101
Human:how many students can enroll in PSY456?
Bot:capacity of PSY456
Human:what's the department of BIO433?
Bot:"""
    num_samples = 1
    api_key = os.environ.get('OPENAI_API_KEY')
    for i in range(num_samples):
        comp_engine = LarkCompletionEngine(college_grammar, 'request', True)
        # rlm = RandomLanguageModel()
        gpt3 = OpenAIModel(model="text-ada-001", prompt_template=college_prompt, api_key=api_key, temperature=1.)
        print(predict_constrained(comp_engine, gpt3, 1, True, stop_tokens=["\n"]))
