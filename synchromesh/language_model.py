#!/usr/bin/env python3

import random
import json
import os
from urllib.request import urlopen
from typing import Optional
import shelve

import openai
import transformers
import torch

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

    def tokenize(self, s: str) -> list[int]:
        return list(map(ord, s))

    def predict_token(self, prefix: str, valid_tokens: list[int], top_k: int = 1) -> tuple[list[int], list[float]]:
        predictions = random.sample(valid_tokens, min(top_k, len(valid_tokens)))
        probabilities = [1.0 / len(predictions)] * len(predictions)
        return predictions, probabilities

    def predict_unconstrained(self, prefix, max_tokens, stop=None):
        return ''.join(random.choices(self.vocabulary(), k=max_tokens)), [0.0]


def download_or_use_cached(url, path):
    if not os.path.exists(path):
        with urlopen(url) as response:
            with open(path, 'wb') as f:
                f.write(response.read())
    return path


class HuggingFaceModel(LanguageModel):
    def __init__(self, model, prompt_template: str, api_key: str = None,
                 temperature: float = 0.0, top_p: float = 1.0, best_of: int = 1,
                 before_prediction_hook=lambda: None, tokenizer=None, device='cuda') -> None:
        super().__init__()

        self.prompt_template = prompt_template
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self.best_of = best_of
        self._before_prediction_hook = before_prediction_hook

        # self.vocab is a list of readable token strings (e.g., ' hello' and '\n')
        # sorted by their token IDs (so self.vocab[0] is the first token, etc).
        self.vocab = [v for k, v in
                      sorted([(t_id, self.tokenizer.decode([t_id]))
                              for _, t_id in self.tokenizer.get_vocab().items()])]

        # HACK: Is there a better way to know if a token has a prefix space?
        # We should only need this for LlamaTokenizer.
        if self.tokenizer.__class__.__name__.startswith('LlamaTokenizer'):
            print('LlamaTokenizer')
            for i in range(len(self.vocab)):
                t = self.vocab[i]
                if 2*len(t) != len(self.tokenizer.decode([i, i], add_special_tokens=False)):
                    self.vocab[i] = ' ' + t
                if t == '':
                    self.vocab[i] = ' '

    def tokenize(self, s: str) -> list[int]:
        return self.tokenizer.encode(s, add_special_tokens=False)

    def vocabulary(self) -> list[str]:
        return self.vocab

    def predict_token(self, prefix: str, valid_tokens: list[int], top_k: int = 1) -> tuple[list[int], list[float]]:
        input_ids = self.tokenizer.encode(prefix, return_tensors="pt", add_special_tokens=False)
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits[:, -1]
            valid_tokens_mask = torch.zeros(logits.shape[-1], dtype=torch.bool)
            valid_tokens_set = set(valid_tokens)
            if None in valid_tokens_set:
                valid_tokens_set.remove(None)
            valid_tokens = list(valid_tokens_set)
            # make sure there's no None in valid_tokens
            valid_tokens_mask[valid_tokens] = True
            logits = logits.squeeze(0)
            filtered_logits = logits.softmax(dim=-1)
            filtered_logits[~valid_tokens_mask] = 0
            filtered_logits = filtered_logits / filtered_logits.sum()

        top_values, top_indices = torch.topk(filtered_logits, top_k)
        top_indices = top_indices.tolist()
        top_values = top_values.log().tolist()
        
        return top_indices, top_values

    def predict_unconstrained(self, prefix: str, max_tokens: int, stop=None):
        prompt = f"{self.prompt_template}{prefix}"
        self._before_prediction_hook()
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=False)
        # Cut off inputs which are too long
        input_length = self.model.config.max_position_embeddings - max_tokens - 1
        if len(input_ids[0]) > input_length:
            input_ids = input_ids[:, -input_length:]
        input_ids = input_ids.to('cuda')
        with torch.cuda.amp.autocast():
            if self.temperature == 0.0:
                output = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            else:
                output = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            # remove the prompt
            output = output[:, len(input_ids[0]):]
        detokenized = self.tokenizer.decode(output[0])
        for stop_token in stop:
            if stop_token in detokenized:
                # split on the first stop token
                detokenized = detokenized.split(stop_token)[0]
        return detokenized

    def predict_constrained_streaming(self,
                                      prefix: str,
                                      constraint_stream: 'StreamingCSD',
                                      max_tokens:int) -> str:
        input_ids = self.tokenizer.encode(f'{self.prompt_template}{prefix}',
                                          return_tensors="pt", add_special_tokens=False)
        input_ids = input_ids.to(self.device)
        past_key_values = None

        with torch.no_grad():
            it = 0
            while it < max_tokens and not constraint_stream.is_complete():
                it += 1
                model_out = self.model(input_ids,
                                       use_cache=True,
                                       past_key_values=past_key_values)

                past_key_values = model_out.past_key_values
                logits = model_out.logits[:, -1].squeeze(0)
                token_p = (logits / self.temperature).softmax(-1)

                # Sample a token and check if valid. If not, compute constraints.
                next_token = token_p.multinomial(1).item()

                if not constraint_stream.can_token_follow(next_token):
                    valid_tokens = constraint_stream.get_valid_tokens()
                    valid_tokens_mask = torch.zeros(logits.shape[-1], dtype=torch.bool)
                    valid_tokens_set = set(valid_tokens)

                    if None in valid_tokens_set:
                        valid_tokens_set.remove(None)

                    valid_tokens = list(valid_tokens_set)
                    valid_tokens_mask[valid_tokens] = True
                    token_p = logits[:]
                    token_p[~valid_tokens_mask] = float('-inf')
                    token_p = (token_p / self.temperature).softmax(-1)

                    # Renormalize and resample
                    assert token_p.sum() > 0, \
                            f"No valid tokens at given prefix '{constraint_stream.get_current_prediction()}'. This might be an issue with the Completion Engine."
                    next_token = token_p.multinomial(1).item()

                    assert next_token in valid_tokens, 'Sampled a forbidden token. This is likely a bug.'

                constraint_stream.feed_prediction(next_token)
                input_ids = torch.ones((1, 1), device=self.device, dtype=int) * next_token

        return constraint_stream.get_current_prediction()


def make_request_key(model, prompt, best_of,
                     max_tokens, temperature, valid_tokens):
    valid_tokens = sorted(valid_tokens) if valid_tokens else None

    kvs = [('model', model), ('prompt', prompt),
           ('best_of', best_of), ('valid_tokens', valid_tokens),
           ('max_tokens', max_tokens), ('temperature', temperature)]

    kvs.sort()
    return ';'.join([f'{repr(k)}={repr(v)}' for k, v in kvs])


class OpenAIModel(LanguageModel):
    def __init__(self, model: str, prompt_template: str, api_key: str = None,
                 temperature: float = 0.0, top_p: float = 1.0, best_of: int = 1,
                 before_prediction_hook=lambda: None,
                 cache_path: Optional[str] = None) -> None:
        super().__init__()

        if api_key:
            openai.api_key = api_key

        self.prompt_template = prompt_template
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.best_of = best_of
        self._before_prediction_hook = before_prediction_hook

        if cache_path:
            self._cache = shelve.open(cache_path)
        else:
            self._cache = {}

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
        # but now we need the model to output the token '0' alone. Here
        # we hit the problem: the sequence of tokens '2' followed by '0'
        # was seen exactly 0 times during training, since the tokenizer
        # will never emit this sequence (it will instead use the '20' token).
        # Hence, the model puts near 0 probability to predicting '0'
        # after predicting '2'. The solution is to only let the model
        # output maximal valid tokens, avoiding this issue.
        valid_tokens = filter_maximal_tokens(valid_tokens, self.tokenizer)

        if len(valid_tokens) == 1:
            return valid_tokens, [0.0]

        # select shortest valid tokens if valid tokens are less than 1200; 4 requests
        if len(valid_tokens) >= 299*4:
            token_lens = [len(self.get_token(i)) for i in valid_tokens]
            # sort valid tokens by length
            valid_tokens = [x for _, x in sorted(zip(token_lens, valid_tokens))]
            valid_tokens = valid_tokens[:299*4-1]

        request_key = make_request_key(self.model, prompt, self.best_of, 1,
                                       self.temperature, valid_tokens)

        if request_key in self._cache:
            return self._cache.get(request_key)  # type: ignore

        for i in range(len(valid_tokens)//299+1):
            valid_bias = {k: 100 for k in valid_tokens[i*299:(i+1)*299]}
            # add a negative bias for the stop token
            valid_bias[50256] = -100
            self._before_prediction_hook()

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

        self._cache[request_key] = (predictions, probabilities)

        return predictions, probabilities

    def predict_unconstrained(self, prefix, max_tokens, stop=None):
        prompt = f"{self.prompt_template}{prefix}"
        self._before_prediction_hook()

        model_limit = get_token_limit(self.model)
        prompt_tokens = len(self.tokenizer.encode(prompt))

        request_key = make_request_key(self.model, prompt, self.best_of, max_tokens,
                                       self.temperature, None)

        if request_key in self._cache:
            return self._cache.get(request_key)

        response = openai.Completion.create(model=self.model, prompt=prompt,
                                            temperature=self.temperature, top_p=self.top_p,
                                            logit_bias={50256: -100},
                                            best_of=self.best_of,
                                            max_tokens=min(
                                                max_tokens,
                                                model_limit - prompt_tokens - 1),
                                            stop=stop)

        self._cache[request_key] = response.choices[0].text

        return response.choices[0].text

# Source: https://platform.openai.com/docs/models/
def get_token_limit(model_name):
    if 'code-davinci' in model_name:
        return 8001
    elif 'code-cushman' in model_name:
        return 2048
    elif 'text-davinci-00' in model_name:
        return 4097
    elif 'curie' in model_name or 'babbage' in model_name or 'ada' in model_name:
        return 2049
    elif 'gpt-4' in model_name:
        return 8192
    elif 'gpt-3.5-turbo' in model_name:
        return 4096
    raise ValueError('Unknown model ' + model_name)


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
