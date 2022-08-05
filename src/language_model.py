#!/usr/bin/env python3

import random
import json
from urllib.request import urlopen

import openai


class LanguageModel:
    def vocabulary(self) -> list[str]:
        raise NotImplementedError()

    def stop_token(self) -> int:
        raise NotImplementedError()

    def predict_token(self, prefix: str, valid_tokens: list[int]) -> int:
        'Given prefix (prompt + already generated code), predicts next token'
        raise NotImplementedError()


class RandomLanguageModel(LanguageModel):
    def vocabulary(self) -> list[str]:
        return list(map(chr, range(128)))

    def predict_token(self, prefix: str, valid_tokens: list[int]) -> int:
        return random.choice(valid_tokens)

    def stop_token(self) -> int:
        return ord('\n')

class OpenAIModel(LanguageModel):
    def __init__(self, model: str, prompt_template: str, api_key: str,
                 temperature: float = 1.0, top_p: float = 1.0, best_of: int = 1) -> None:
        super().__init__()
        openai.api_key = api_key
        self.prompt_template = prompt_template
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.best_of = best_of
        # for gpt series of models
        url = "https://huggingface.co/gpt2/resolve/main/vocab.json"
        response = urlopen(url)
        self.token_idx = json.loads(response.read())
        self.token_idx = {s.replace('\u0120',' '):i for s,i in self.token_idx.items()}
            

    def vocabulary(self) -> list[str]:
        # sort keys by value, then return the keys
        vocab = sorted(self.token_idx.keys(), key=lambda k: self.token_idx[k])
        return vocab

    def predict_token(self, prefix: str, valid_tokens: list[int]) -> int:
        # change bias of valid tokens to make them more likely
        # bias can only be set for 300 tokens
        # hacky way to do this
        # TODO: change to use masking over logits instead of bias
        candidates = []
        prompt = f"{self.prompt_template}{prefix}"
        for i in range(len(valid_tokens)//300+1):
            valid_bias = {k: 100 for k in valid_tokens[i*300:(i+1)*300]}
            response = openai.Completion.create(model=self.model, prompt=prompt,
                                                temperature=self.temperature, top_p=self.top_p,
                                                best_of=self.best_of, max_tokens=1, logit_bias=valid_bias)
            candidates.append(self.token_idx[response.choices[0].text])
        if len(candidates) > 1:
            valid_bias = {k: 100 for k in candidates}
            response = openai.Completion.create(model=self.model, prompt=prompt,
                                                temperature=self.temperature, top_p=self.top_p,
                                                best_of=self.best_of, max_tokens=1, logit_bias=valid_bias)

        return self.token_idx[response.choices[0].text]

    def stop_token(self) -> int:
        return "<eos"
