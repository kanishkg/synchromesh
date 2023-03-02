#!/usr/bin/env python3

import random
import json
import os
from urllib.request import urlopen

import openai
import transformers


class LanguageModel:
    def vocabulary(self) -> list[str]:
        raise NotImplementedError()

    def predict_tokens(self, prefix: str, n: int) -> list[int]:
        raise NotImplementedError()

    def predict_token(self, prefix: str, valid_tokens: list[int], top_k: int = 1) -> tuple[list[int], list[float]]:
        'Given prefix (prompt + already generated code), predicts next token'
        raise NotImplementedError()

    def tokenize(self, s: str) -> list[int]:
        raise NotImplementedError()

    def get_token(self, i: int) -> str:
        return self.vocabulary()[i]

    def predict_unconstrained(self, prefix: str, max_tokens: int, stop=None):
        raise NotImplementedError()


class RandomLanguageModel(LanguageModel):
    def vocabulary(self) -> list[str]:
        return list(map(chr, range(128)))

    def predict_token(self, prefix: str, valid_tokens: list[int], top_k: int = 1) -> tuple[list[int], list[float]]:
        predictions = random.sample(valid_tokens, min(top_k, len(valid_tokens)))
        probabilities = [1.0 / len(predictions)] * len(predictions)
        return predictions, probabilities

    def predict_unconstrained(self, prefix, max_tokens, stop=None):
        return ''.join(random.choices(self.vocabulary(), k=max_tokens))


def download_or_use_cached(url, path):
    if not os.path.exists(path):
        with urlopen(url) as response:
            with open(path, 'wb') as f:
                f.write(response.read())
    return path

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
            self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        elif model.startswith("code"):
            self.tokenizer = transformers.GPT2Tokenizer(
                vocab_file=download_or_use_cached(
                    "https://huggingface.co/SaulLu/codex-like-tokenizer/raw/main/vocab.json",
                    '.codex-vocab.json'),
                merges_file=download_or_use_cached(
                    "https://huggingface.co/SaulLu/codex-like-tokenizer/raw/main/merges.txt",
                    '.codex-merges.txt')
                )

        # self.vocab is a list of readable token strings (e.g., ' hello' and '\n')
        # sorted by their token IDs (so self.vocab[0] is the first token, etc).
        self.vocab = [v for k, v in
                      sorted([(t_id, self.tokenizer.decode([t_id]))
                              for _, t_id in self.tokenizer.get_vocab().items()])]

    def tokenize(self, s: str) -> list[int]:
        return self.tokenizer.encode(s)

    def vocabulary(self) -> list[str]:
        # sort keys by value, then return the keys
        return self.vocab

    def predict_token(self, prefix: str, valid_tokens: list[int], top_k: int = 1) -> tuple[list[int], list[float]]:
        # change bias of valid tokens to make them more likely
        # bias can only be set for 300 tokens at a time
        assert top_k <= 5, "top_k must be less than or equal to 5"
        predictions, probabilities = [], []
        prompt = f"{self.prompt_template}{prefix}"

        # Only keep tokens that cannot be extended. This is crucial, because
        # the model has *never* seen a sequence of non-maximal tokens in its
        # input, and if we force it to output a sequence of maximal tokens,
        # a logit bias is often not enough to constrain it (it outputs near
        # zero probability for the valid tokens even after adding 100 to
        # the logits).
        #
        # Longer explanation:
        # Suppose we want to force the model to output
        # the number 20302. This would be BPE-tokenized as 20 | 302.
        # Suppose we let the model output '2' alone. We succeed at that,
        # but not we need the model to output the token '0' alone. This
        # is the problem: '2' and '0' were seen exactly 0 times during
        # training, since the tokenizer will never emit this sequence of
        # tokens. Hence, the model puts near 0 probability to predicting '0'
        # after predicting '2'. By not letting it output non-maximal tokens
        # in the first place, we avoid this issue.
        valid_tokens = filter_maximal_tokens(valid_tokens, self.tokenizer)

        if len(valid_tokens) == 1:
            return valid_tokens, [0.0]

        # select shortest valid tokens if valid tokens are less than 1200; 4 requests
        if len(valid_tokens) >= 299*4:
            token_lens = [len(self.get_token(i)) for i in valid_tokens]
            # sort valid tokens by length
            valid_tokens = [x for _, x in sorted(zip(token_lens, valid_tokens))]
            valid_tokens = valid_tokens[:299*4-1]

        for i in range(len(valid_tokens)//299+1):
            valid_bias = {k: 100 for k in valid_tokens[i*299:(i+1)*299]}
            # add a negative bias for the stop token
            valid_bias[50256] = -100
            # TODO: Using codex leads to a bug
            response = openai.Completion.create(model=self.model, prompt=prompt, logprobs=top_k,
                                                temperature=self.temperature, top_p=self.top_p,
                                                best_of=self.best_of, max_tokens=1, logit_bias=valid_bias)

            response_dict = response.choices[0].logprobs.top_logprobs[0]

            for k in sorted(response_dict.keys()):
                predictions.append(self.tokenizer.encode(k)[0])
                probabilities.append(response_dict[k])

        # sort predictions by probability
        predictions = [c for _, c in sorted(zip(probabilities, predictions), key=lambda x: x[0], reverse=True)]
        probabilities = sorted(probabilities, reverse=True)
        predictions = predictions[:min(top_k, len(predictions))]
        predictions = list(predictions)
        probabilities = probabilities[:min(top_k, len(probabilities))]
        breakpoint()
        return predictions, probabilities

    def predict_unconstrained(self, prefix, max_tokens, stop=None):
        prompt = f"{self.prompt_template}{prefix}"
        response = openai.Completion.create(model=self.model, prompt=prompt,
                                            temperature=self.temperature, top_p=self.top_p,
                                            logit_bias={50256: -100},
                                            best_of=self.best_of, max_tokens=max_tokens, stop=stop)
        return response.choices[0].text


def filter_maximal_tokens(tokens: list[int], tokenizer) -> list[int]:
    '''Given a list of tokens, only keep the maximal ones.

    This takes quadratic time; might be slow with overly long lists.
    NOTE: This can be made linear-time by inserting all tokens in a Trie
    and then only taking the leaves.
    '''

    token_strs = list(map(tokenizer.decode, tokens))
    result = []

    for i in range(len(tokens)):
        is_maximal = True

        for j in range(len(tokens)):
            if i != j and token_strs[j].startswith(token_strs[i]):
                is_maximal = False
                break

        if is_maximal:
            result.append(tokens[i])

    return result
