# -*- coding: utf-8-*-
import os
import pickle
import random
from argparse import ArgumentParser

from feature_extractor import FeatureExtractor
from utils import set_random_seed, str2bool


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_paths', type=str, default='')
    parser.add_argument('--valid_paths', type=str, default='')
    parser.add_argument('--test_paths', type=str, default='')
    parser.add_argument('--separator', choices=['whitespace', 'tab'], default='whitespace')
    parser.add_argument('--train_rate', type=int, default=8)
    parser.add_argument('--valid_rate', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--rand_seed', type=int, default=777)
    parser.add_argument('--do_lowercase', type=str2bool, nargs='?', default=True)

    args = parser.parse_args()
    set_random_seed(args.rand_seed)

    os.makedirs(args.output_dir, exist_ok=True)

    feature_extractor = FeatureExtractor(do_lowercase=args.do_lowercase)

    train_paths = args.train_paths.split(',')
    valid_paths = args.valid_paths.split(',')
    test_paths = args.test_paths.split(',')
    if args.separator == 'whitespace':
        separator = ' '
    elif args.separator == 'tab':
        separator = '\t'
    else:
        raise Exception('unknown separator type')
    train_examples = feature_extractor.run(train_paths, separator=separator, for_train=True)
    if valid_paths:
        valid_examples = feature_extractor.run(valid_paths, separator=separator, for_train=False)
    else:
        split_index = int(len(train_examples) * args.train_rate / (args.train_rate + args.valid_rate))
        random.shuffle(train_examples)
        valid_examples = train_examples[split_index:]
        train_examples = train_examples[:split_index]
    test_examples = feature_extractor.run(test_paths, separator=separator, for_train=False)

    with open(os.path.join(args.output_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train_examples, f)
    with open(os.path.join(args.output_dir, 'valid.pkl'), 'wb') as f:
        pickle.dump(valid_examples, f)
    with open(os.path.join(args.output_dir, 'test.pkl'), 'wb') as f:
        pickle.dump(test_examples, f)

    vocab = feature_extractor.vocab()
    with open(os.path.join(args.output_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)


if __name__ == '__main__':
    main()
