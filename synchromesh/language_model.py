#!/usr/bin/env python3

import random
import json
from urllib.request import urlopen

import openai
from typing import List
from typing import Tuple
from typing import Dict
import time
import os


class LanguageModel:
    def vocabulary(self) -> List[str]:
        raise NotImplementedError()

    def predict_tokens(self, prefix: str, n: int) -> List[int]:
        raise NotImplementedError()

    def predict_token(self, prefix: str, valid_tokens: List[int], top_k: int = 1) -> Tuple[List[int], List[float]]:
        'Given prefix (prompt + already generated code), predicts next token'
        raise NotImplementedError()

    def tokenize(self, s: str) -> List[int]:
        raise NotImplementedError()

    def get_token(self, i: int) -> str:
        return self.vocabulary()[i]

    def predict_unconstrained(self, prefix: str, max_tokens: int, stop=None):
        raise NotImplementedError()


class RandomLanguageModel(LanguageModel):
    def vocabulary(self) -> List[str]:
        return List(map(chr, range(128)))

    def predict_token(self, prefix: str, valid_tokens: List[int], top_k: int = 1) -> Tuple[List[int], List[float]]:
        predictions = random.sample(valid_tokens, min(top_k, len(valid_tokens)))
        probabilities = [1.0 / len(predictions)] * len(predictions)
        return predictions, probabilities

    def predict_unconstrained(self, prefix, max_tokens, stop=None):
        return ''.join(random.choices(self.vocabulary(), k=max_tokens))


class OpenAIModel(LanguageModel):
    def __init__(self, model: str, prompt_template: str, api_key: str,
                 temperature: float = 0.0, top_p: float = 1.0, best_of: int = 1) -> None:
        super().__init__()
        openai.api_key = api_key
        self.prompt_template = prompt_template
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.best_of = best_of
        # for gpt series of models
        if model.startswith("text"):
            url = "https://huggingface.co/gpt2/resolve/main/vocab.json"
            with urlopen(url) as response:
                self.token_idx = json.loads(response.read())
            self.token_idx = {s.replace('\u0120', ' '): i
                              for s, i in self.token_idx.items()}
            self.vocab = sorted(self.token_idx.keys(), key=lambda k: self.token_idx[k])
        elif model.startswith("code"):
            url = "https://huggingface.co/SaulLu/codex-like-tokenizer/raw/main/vocab.json"
            with urlopen(url) as response:
                self.token_idx = json.loads(response.read())
            self.token_idx = {s.replace('\u0120', ' '): i
                              for s, i in self.token_idx.items()}
            self.vocab = sorted(self.token_idx.keys(), key=lambda k: self.token_idx[k])

    def tokenize(self, s: str) -> List[int]:
        vocab = self.vocabulary()
        tokens = []

        while s:
            # Find longest token that is a prefix of s.
            l = 1
            while l <= len(s) and s[:l] in self.token_idx:
                l += 1
            # Add it to tokens, remove from s.
            tokens.append(self.token_idx[s[:(l - 1)]])
            s = s[(l - 1):]

        return tokens

    def vocabulary(self) -> List[str]:
        # sort keys by value, then return the keys
        return self.vocab

    def predict_token(self, prefix: str, valid_tokens: List[int], top_k: int = 1) -> Tuple[List[int], List[float]]:
        # change bias of valid tokens to make them more likely
        # bias can only be set for 300 tokens at a time
        assert top_k <= 5, "top_k must be less than or equal to 5"
        predictions, probabilities = [], []
        prompt = f"{self.prompt_template}{prefix}"

        # select shortest valid tokens if valid tokens are less than 1200; 4 requests
        if len(valid_tokens) >= 299 * 4:
            token_lens = [len(self.get_token(i)) for i in valid_tokens]
            # sort valid tokens by length
            valid_tokens = [x for _, x in sorted(zip(token_lens, valid_tokens))]
            valid_tokens = valid_tokens[:299 * 4 - 1]

        for i in range(len(valid_tokens) // 299 + 1):
            valid_bias = {k: 100 for k in valid_tokens[i * 299:(i + 1) * 299]}
            # add a negative bias for the stop token
            valid_bias[50256] = -100
            # TODO: Using codex leads to a bug
            response = openai.Completion.create(model=self.model, prompt=prompt, logprobs=top_k,
                                                temperature=self.temperature, top_p=self.top_p,
                                                best_of=self.best_of, max_tokens=1, logit_bias=valid_bias)
            response_dict = response.choices[0].logprobs.top_logprobs[0]
            for k in sorted(response_dict.keys()):
                predictions.append(self.token_idx[k])
                probabilities.append(response_dict[k])

        # sort predictions by probability
        predictions = [c for _, c in sorted(zip(probabilities, predictions), key=lambda x: x[0], reverse=True)]
        probabilities = sorted(probabilities, reverse=True)
        predictions = predictions[:min(top_k, len(predictions))]
        predictions = [c for c in predictions]
        probabilities = probabilities[:min(top_k, len(probabilities))]
        return predictions, probabilities

    def predict_unconstrained(self, prefix, max_tokens, stop=None, mocked=False, randomized=False):
        time.sleep(0.5)
        if mocked:
            return self.predict_mock(randomized)

        prompt = f"{self.prompt_template}{prefix}"
        response = openai.Completion.create(model=self.model, prompt=prompt,
                                            temperature=self.temperature, top_p=self.top_p,
                                            best_of=self.best_of, max_tokens=max_tokens, stop=stop)
        return response.choices[0].text

    def predict_mock(self, randomized=False, seed=100):
        gold_output_files = os.listdir('gold')
        if randomized:
            random.seed(seed)
            output = random.choice(gold_output_files)
        else:
            output = gold_output_files[0]
        output = "./gold/" + output
        with open(output, 'r') as f:
            return f.read()
