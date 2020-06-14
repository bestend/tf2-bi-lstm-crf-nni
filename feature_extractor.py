# -*- coding: utf-8-*-
from typing import List, Dict

import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

S_PAD = '[PAD]'
S_UNK = '[UNK]'
V_PAD = 0
V_UNK = 1


class Vocabulary(BaseModel):
    token: Dict[str, int] = {}
    char: Dict[str, int] = {}
    pos: Dict[str, int] = {}
    label: Dict[str, int] = {}

    def idx_to_label(self):
        return {v: k for k, v in self.label.items()}


class Example(BaseModel):
    raw_token: List[str] = []
    token: List[int] = []
    char: List[List[int]] = []
    pos: List[int] = []
    label: List[int] = []


class FeatureExtractor(object):
    FEATURES = ['token', 'char', 'pos', 'label']

    def __init__(self, vocab=None, do_lowercase=True):
        self._vocab = vocab if vocab else Vocabulary()
        self._vocab = {name: {S_PAD: V_PAD, S_UNK: V_UNK} for name in self.FEATURES}
        self._vocab['label']['O'] = len(self._vocab['label'])
        self._do_lowercase = do_lowercase

    def vocab(self) -> Vocabulary:
        return Vocabulary(**self._vocab)

    def run(self, paths: List[str], separator, for_train=False) -> List[Example]:
        examples: List[Example] = []
        for path in tqdm(paths, desc='file'):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    raw_example = {name: [] for name in self.FEATURES if name != 'char'}
                    for idx, line in enumerate(f):
                        if idx == 0 and line.startswith('-DOCSTART-'):
                            continue
                        line = line.strip('\n')
                        if line:
                            tokens = line.split(separator)
                            token = tokens[0]
                            pos = tokens[1]
                            label = tokens[-1]
                            raw_example['token'].append(token)
                            raw_example['pos'].append(pos)
                            raw_example['label'].append(label)
                        else:
                            examples.append(self._convert_example(raw_example, for_train))
                            raw_example = {name: [] for name in self.FEATURES if name != 'char'}
                    if raw_example:
                        examples.append(self._convert_example(raw_example, for_train))
            except Exception as e:
                print(f"{path}:{idx} 에 오류가 있습니다. = {str(e)}")
                raise e
        print(f"total examples: {len(examples)}")
        print(f"label num: {len(self._vocab['label'])}")
        print(f"pos num: {len(self._vocab['pos'])}")
        print(f"max token length: {np.max([len(example.token) for example in examples])}")
        print(f"avg token length: {np.mean([len(example.token) for example in examples])}")
        print(f"max char length: {np.max([len(c) for example in examples for c in example.char])}")
        print(f"avg char length: {np.mean([len(c) for example in examples for c in example.char])}")
        return examples

    def _preprocess(self, token):
        if self._do_lowercase:
            return token.lower()
        else:
            return token

    def _convert_token(self, elements, vocab, for_train=False, ignore_case=False):
        values = []
        for e in elements:
            if ignore_case:
                e = e.lower()
            if for_train is True and e not in vocab:
                vocab[e] = len(vocab)
            value = vocab.get(e, V_UNK)
            values.append(value)
        return values

    def _convert_example(self, raw_example: Dict[str, List[str]], for_train: bool = False) -> Example:
        example = {}
        example['raw_token'] = raw_example['token']
        raw_example['token'] = [self._preprocess(t) for t in raw_example['token']]
        for name in self.FEATURES:
            if name == 'char':
                example[name] = [self._convert_token(t, self._vocab[name], for_train) for t in
                                 example['raw_token']]
            else:
                example[name] = self._convert_token(raw_example[name], self._vocab[name], for_train)
        return Example(**example)
