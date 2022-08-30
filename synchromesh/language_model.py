#!/usr/bin/env python3

import random
import json
from urllib.request import urlopen

import openai


class LanguageModel:
    def vocabulary(self) -> list[str]:
        raise NotImplementedError()

    def predict_token(self, prefix: str, valid_tokens: list[int], top_k: int = 1) -> tuple[list[int], list[float]]:
        'Given prefix (prompt + already generated code), predicts next token'
        raise NotImplementedError()


class RandomLanguageModel(LanguageModel):
    def vocabulary(self) -> list[str]:
        return list(map(chr, range(128)))

    def predict_token(self, prefix: str, valid_tokens: list[int], top_k: int = 1) -> tuple[list[int], list[float]]:
        predictions = random.sample(valid_tokens, min(top_k, len(valid_tokens)))
        probabilities = [1.0 / len(predictions)] * len(predictions)
        return predictions, probabilities

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

    def predict_token(self, prefix: str, valid_tokens: list[int], top_k: int = 1) -> tuple[list[int], list[float]]:
        # change bias of valid tokens to make them more likely
        # bias can only be set for 300 tokens at a time
        assert top_k <= 5, "top_k must be less than or equal to 5"
        predictions, probabilities = [], []
        prompt = f"{self.prompt_template}{prefix}"
        for i in range(len(valid_tokens)//300+1):
            valid_bias = {k: 100 for k in valid_tokens[i*300:(i+1)*300]}
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
        return  predictions, probabilities
    
    def predict_unconstrained_token(self, prompt, max_tokens, stop=None):
        response = openai.Completion.create(model=self.model, prompt=prompt, logprobs=5,
                                            temperature=self.temperature, top_p=self.top_p,
                                            best_of=self.best_of, max_tokens=max_tokens, stop=stop)
        response_dict = response.choices[0].logprobs.top_logprobs[0]
        text = response.choices[0].text
        return text
