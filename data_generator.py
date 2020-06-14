# -*- coding: utf-8-*-
import copy
import pickle
from typing import List

import numpy as np
import tensorflow as tf
from math import ceil
from tensorflow.keras.utils import Sequence

from feature_extractor import Vocabulary, Example
from utils import clip_pad


class DataGenerator(Sequence):
    def __init__(self, paths: List[str], vocab: Vocabulary, max_word: int, max_char: int, batch_size: int,
                 shuffle: bool = True, elmo=None) -> None:
        self._vocab = vocab
        self._data = self._read_data(paths)
        self._data_size = 1000
        self._batch_size = batch_size
        self._max_word = max_word
        self._max_char = max_char
        self._shuffle = shuffle
        self._elmo = elmo

        self.on_epoch_end()

    def _read_data(self, paths) -> List[Example]:
        examples = []
        for path in paths:
            with open(path, 'rb') as f:
                current = pickle.load(f)
                examples.extend(current)
        return examples

    def __len__(self) -> int:
        return int(ceil(self._data_size / self._batch_size))

    def __getitem__(self, index):
        inputs = {
            "raw_token": [],
            "token": [],
            "char": [],
            "pos": [],
        }
        outputs = []

        begin = index * self._batch_size
        end = min(self._data_size, (index + 1) * self._batch_size)
        max_word = min(self._max_word, max([len(example.token) for example in self._data[begin:end]]))
        inputs['sequence_lengths'] = [min(max_word, len(example.token)) for example in self._data[begin:end]]

        for example in self._data[begin:end]:
            example = copy.deepcopy(example)
            inputs["raw_token"].append(clip_pad(example.raw_token, max_word, [""]))
            inputs["token"].append(clip_pad(example.token, max_word))
            for idx, c in enumerate(example.char):
                example.char[idx] = clip_pad(c, self._max_char)
            inputs["char"].append(clip_pad(example.char, max_word, [[0] * self._max_char]))
            inputs["pos"].append(clip_pad(example.pos, max_word))
            outputs.append(clip_pad(example.label, max_word))

        inputs = {name: np.asarray(x) for name, x in inputs.items()}
        if self._elmo:
            inputs['token'] = self._elmo.signatures['tokens'](
                tokens=tf.squeeze(tf.cast(inputs['raw_token'], tf.string)),
                sequence_len=tf.cast(inputs['sequence_lengths'], tf.int32)
            )["elmo"]
        outputs = np.asarray(outputs)
        return inputs, outputs

    def on_epoch_end(self):
        if self._shuffle:
            np.random.shuffle(self._data)
