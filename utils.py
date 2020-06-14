# -*- coding: utf-8-*-
import argparse
import os
import pickle
from collections import OrderedDict
from datetime import datetime

from ruamel import yaml

from feature_extractor import Vocabulary
from train_param import TrainParam


def load_param(conf: str) -> TrainParam:
    with open(conf, 'r', encoding='utf-8') as f:
        raw_param = yaml.safe_load(f)
        return TrainParam(**raw_param)


def load_vocabulary(path: str) -> Vocabulary:
    with open(path, 'rb') as f:
        return pickle.load(f)


def set_random_seed(seed_value: int = -1) -> None:
    if seed_value < 0:
        seed_value = datetime.now()

    os.environ['PYTHONHASHSEED'] = str(seed_value)
    import random
    random.seed(seed_value)
    import numpy as np
    np.random.seed(seed_value)
    import tensorflow as tf
    tf.random.set_seed(seed_value)
    import torch
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def gpu_config(use_gpu: bool = True) -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if use_gpu:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
    else:
        import tensorflow as tf
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    tf.keras.backend.set_floatx('float32')
    tf.summary.trace_on(graph=True)


def clip_pad(data, length, pad=[0]):
    if len(data) < length:
        data += pad * (length - len(data))
    return data[:length]


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _sort_data(od):
    if isinstance(od, str):
        return od

    res = OrderedDict()
    for k, v in sorted(od.items()):
        if isinstance(v, dict):
            res[k] = _sort_data(v)
        elif isinstance(v, list):
            res[k] = [_sort_data(e) for e in v]
        else:
            res[k] = v
    return res
