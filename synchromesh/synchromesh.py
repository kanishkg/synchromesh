#!/usr/bin/env python3

import os
import regex
import time

from .completion_engine import CompletionEngine, LarkCompletionEngine
from .language_model import LanguageModel, RandomLanguageModel, OpenAIModel
from . import trie


class StreamingCSD:
    '''Streaming implementation of Constrained Semantic Decoding

    Use this if you want full control when sampling from the model
    (e.g., if you're using Hugging Face models, not OpenAI).

    This is the suggested approach:

    While not done sampling:
    - Predict distribution over next token from the language model.
    - Sample a token.
    - Check if can_token_follow returns True.
    -- If so, call feed_prediction and continue.
    -- Otherwise, call get_valid_tokens(), sample from that support set,
       then feed_prediction and continue.

    The reason for this is that can_token_follow is more efficient
    than get_valid_tokens (which iterates over the vocabulary, although
    with very heavy pruning). Thus, the fewer calls to get_valid_tokens()
    you can do, the better.
    '''

    def __init__(self,
                 completion_engine: CompletionEngine,
                 lm_vocabulary: list[str]):
        self._trie = trie.Trie.from_vocabulary(lm_vocabulary)
        self._vocab = lm_vocabulary
        self._completion_engine = completion_engine
        self._completion_points: dict[str, regex.Pattern] = {}
        self._completion_points[''] = completion_engine.complete('')

        self.init_stream()

    def init_stream(self):
        self._prefix_tokens = []
        self._prefix_str = ''

    def can_token_follow(self, t: int):
        return is_prefix_valid(self._completion_engine,
                               self._completion_points,
                               self._prefix_str + self._vocab[t])

    def feed_prediction(self, t: int):
        self._prefix_tokens.append(t)
        self._prefix_str += self._vocab[t]

    def get_valid_tokens(self) -> list[int]:
        return self._trie.antimonotonic_filter(
                lambda t:
                    is_prefix_valid(self._completion_engine,
                        self._completion_points,
                        self._prefix_str + t))

    def get_current_prediction(self) -> str:
        return self._prefix_str

    def get_current_prediction_tokens(self) -> list[int]:
        return self._prefix_tokens

    def is_complete(self) -> bool:
        return self._completion_engine.is_complete(self._prefix_str)

    def fast_forward(self):
        while not self.is_complete():
            v = self.get_valid_tokens()
            if len(v) > 1:
                break
            self.feed_prediction(v[0])


# Implements the Constrained Semantic Decoding algorithm.
def predict_constrained(completion_engine: CompletionEngine, lm: LanguageModel,
                        top_k: int = 1, verbose: bool = False,
                        batch_size: int = 50, stop_tokens: list[str]=None,
                        max_violations: int = 20,
                        fast_forward: bool = False) -> str:

    # If model implements (faster) streaming inference, use that instead of batch + rejection.
    if hasattr(lm, 'predict_constrained_streaming'):
        return lm.predict_constrained_streaming(
                '',
                StreamingCSD(completion_engine, lm.vocabulary()),
                1000
                )

    completion_points: dict[str, regex.Pattern] = {}

    completion_points[''] = completion_engine.complete('')

    token_trie = trie.Trie.from_vocabulary(lm.vocabulary())

    prediction = ''
    n_violations = 0

    while not completion_engine.is_complete(prediction):
        # Ask for unconstrained prediction.
        if verbose:
            print('Prefix:', prediction)

        if fast_forward:
            # While there's only one valid token at this point, simply add
            # that token instead of querying the model. This can be slow but can also
            # save many calls to the model in use cases where the completion engine can
            # output very long constraints (e.g. only let the model choose between generating
            # two long sequences, so after it starts to output one the rest is determined).
            while True:
                valid_tokens = token_trie.antimonotonic_filter(
                    lambda t: is_prefix_valid(completion_engine,
                                              completion_points,
                                              prediction + t)
                )

                if len(valid_tokens) == 1:
                    prediction += lm.get_token(valid_tokens[0])
                    if completion_engine.is_complete(prediction):
                        return prediction
                else:
                    break

            if verbose:
                print('After fast forwarding:', prediction)

        continuation = lm.predict_unconstrained(prediction, batch_size, stop=stop_tokens)
        found_violation = False

        if verbose:
            print('Continuation:', continuation)

        if not continuation:
            # HACK: LM really thinks is done. This will not make progress.
            # Trusting it for now.
            if verbose:
                print('Empty continuation. Stopping early because model refuses to keep going.')
            break

        for token in lm.tokenize(continuation):
            if is_prefix_valid(completion_engine, completion_points,
                               prediction + lm.get_token(token)):
                prediction += lm.get_token(token)
            else:
                if completion_engine.is_complete(prediction):
                    break
                found_violation = True
                if verbose:
                    print(f"Found violation at token: {repr(lm.get_token(token))}")
                    print(f"Valid prefix: {prediction}")
                break

        if found_violation:
            n_violations += 1

            if n_violations > max_violations:
                break

            # Do constrained prediction for next token.
            if verbose:
                print(f"Constrained prediction for: {prediction}")
                print('Determining valid tokens...')

            valid_tokens = token_trie.antimonotonic_filter(
                lambda t: is_prefix_valid(completion_engine,
                                          completion_points,
                                          prediction + t)
            )

            if verbose:
                print('Done:', len(valid_tokens), 'tokens.')

            assert len(valid_tokens) > 0, f"No valid tokens after {repr(prediction)}"
            predictions, probabilities = lm.predict_token(prediction,
                                                          valid_tokens,
                                                          top_k)

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
            if i and completion_point_regex.fullmatch(remainder[:i]):
                # We found another completion point, reduce the problem and call recursively.
                new_completion_point = s[:longest_completion_point] + remainder[:i]
                new_completion_point_regex = completion_engine.complete(new_completion_point)
                completion_points[new_completion_point] = new_completion_point_regex
                return is_prefix_valid(completion_engine, completion_points, s)
            else:
                return False

    #    Case c- Got to the end with no violations, return True
    return True


def test_streaming_csd():
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

        string : "\"" /Some long string here that is fixed/ "\""

        %import common.SIGNED_NUMBER
        """

    comp_engine = LarkCompletionEngine(json_grammar, 'dict', False)
    lm = RandomLanguageModel()

    csd = StreamingCSD(comp_engine, lm.vocabulary())

    import time
    start_time = time.time()

    while not comp_engine.is_complete(csd.get_current_prediction()):
        continuation, _ = lm.predict_unconstrained(csd.get_current_prediction(),
                                                   max_tokens=1)
        tokens = lm.tokenize(continuation)

        if csd.can_token_follow(tokens[0]):
            csd.feed_prediction(tokens[0])
        else:
            valid_tokens = csd.get_valid_tokens()
            tokens, _ = lm.predict_token(csd.get_current_prediction(),
                                         valid_tokens)
            csd.feed_prediction(tokens[0])

        s = csd.get_current_prediction()

        if len(s) > 500:
            break

        csd.fast_forward()

    delta = time.time() - start_time

    print('Predicted:', repr(csd.get_current_prediction()))
    print('Throughput:', len(csd.get_current_prediction_tokens()) / delta, 'tokens/s')

def test_college_grammar_csd():
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


def test_fast_forward():
    fixed_response = r"""
        ?response: "the answer is abcdefghijklmnopqrstuvwxyz"
    """

    prompt = """You are a helpful assistant.

Human: What day is today?
Assistant: Thursday

Human: What is the answer?
Assistant:"""

    num_samples = 1
    api_key = os.environ.get('OPENAI_API_KEY')
    for i in range(num_samples):
        comp_engine = LarkCompletionEngine(fixed_response, 'response', False)

    ada = OpenAIModel(model="text-ada-001", prompt_template=prompt,
                      api_key=api_key, temperature=1.)
    print(predict_constrained(comp_engine, ada, 1, True,
                              stop_tokens=["\n"], fast_forward=True))



if __name__ == '__main__':
    test_fast_forward()
