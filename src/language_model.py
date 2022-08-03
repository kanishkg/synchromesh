#!/usr/bin/env python3

import random


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
