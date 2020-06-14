# -*- coding: utf-8-*-
import logging
import os
import shutil
import time
from argparse import ArgumentParser

import nni
import tensorflow as tf
import tensorflow_hub as hub
from devtools import debug
from ruamel import yaml
from tensorflow_addons.text import crf_decode

import conlleval
from data_generator import DataGenerator
from model import BiLstmCrfModel
from train_param import TrainParam
from utils import load_param, load_vocabulary, set_random_seed, gpu_config, str2bool

logger = logging.getLogger('ner')
logger.setLevel(logging.INFO)


# tf.config.experimental_run_functions_eagerly(True)


@tf.function(experimental_relax_shapes=True)
def train_step(model, optimizer, inputs, outputs):
    with tf.GradientTape() as tape:
        logits, log_likelihood = model(inputs, outputs, training=True)
        loss = - tf.reduce_mean(log_likelihood)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    predicts, _ = crf_decode(logits, model.transition_params, inputs['sequence_lengths'])
    return loss


@tf.function(experimental_relax_shapes=True)
def test_step(model, inputs):
    logits = model(inputs, training=False)
    predicts, _ = crf_decode(logits, model.transition_params, inputs['sequence_lengths'])
    return predicts


def test_loop(idx_to_label, model, output_path, generator):
    total_trues = []
    total_preds = []
    with open(output_path, "w", encoding='utf-8') as f:
        for inputs, outputs in generator:
            predicts = test_step(model, inputs)
            predicts = predicts.numpy()
            for idx, predict in enumerate(predicts):
                output = outputs[idx]
                cur_len = inputs['sequence_lengths'][idx]
                for token_id in range(cur_len):
                    token = inputs["raw_token"][idx][token_id]
                    tr = idx_to_label[output[token_id]]
                    pr = idx_to_label[predict[token_id]]
                    total_trues.append(tr)
                    total_preds.append(pr)
                    f.write(f'{token}\t{tr}\t{pr}\n')
                f.write('\n')
        generator.on_epoch_end()
    prec, rec, f1 = conlleval.evaluate(total_trues, total_preds, verbose=False)
    return f1


def recursive_generator(generator):
    while True:
        for inputs, outputs in generator:
            yield inputs, outputs
        generator.on_epoch_end()


def main(train_param: TrainParam, do_nni: bool) -> None:
    # gpu setting
    set_random_seed(train_param.rand_seed)
    gpu_config()

    # load data
    debug(train_param)
    vocab = load_vocabulary(train_param.vocab_path)

    idx_to_label = vocab.idx_to_label()
    if train_param.use_elmo:
        elmo = hub.load("https://tfhub.dev/google/elmo/3")
    else:
        elmo = None
    train_generator = DataGenerator(train_param.train_paths, vocab, train_param.max_word, train_param.max_char,
                                    train_param.batch_size, shuffle=True, elmo=elmo)
    valid_generator = DataGenerator(train_param.valid_paths, vocab, train_param.max_word, train_param.max_char,
                                    train_param.batch_size, shuffle=False, elmo=elmo)
    if train_param.test_paths:
        test_generator = DataGenerator(train_param.test_paths, vocab, train_param.max_word, train_param.max_char,
                                       train_param.batch_size, shuffle=False, elmo=elmo)

    # model
    model = BiLstmCrfModel(train_param, vocab)
    if not train_param.use_elmo and train_param.pretrained_embedding_path:
        model.load_embedding_weights(train_param.pretrained_embedding_path, vocab)
    tb_callback = tf.keras.callbacks.TensorBoard(train_param.output_dir, profile_batch=0)
    tb_callback.set_model(model)

    # optimizer
    if train_param.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(train_param.learning_rate, decay=train_param.lr_decay_rate,
                                             **train_param.optimizer_param)
    elif train_param.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(train_param.learning_rate, decay=train_param.lr_decay_rate,
                                            **train_param.optimizer_param)
    else:
        raise Exception('unknown optimizer type')

    # checkpoint loader
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt.restore(tf.train.latest_checkpoint(train_param.output_dir))
    ckpt_manager = tf.train.CheckpointManager(ckpt, train_param.output_dir, checkpoint_name='model.ckpt', max_to_keep=3)

    # main
    no_update = 0
    best_score = 0.0
    last_test_score = 0.0
    best_epoch = 0

    generator = recursive_generator(train_generator)
    with tf.summary.create_file_writer(train_param.output_dir + "/train").as_default() as writer:
        for epoch in range(train_param.epochs):
            train_loss = 0.0
            num_batches = 0

            # train loop
            start = time.clock()
            for step in range(train_param.train_steps):
                inputs, outputs = next(generator)
                train_loss += train_step(model, optimizer, inputs, outputs)
                num_batches += 1

                tf.summary.scalar('loss', train_loss / num_batches, step=(epoch * train_param.train_steps + step))
                writer.flush()
                if step % 10 == 0:
                    time_per_step = (time.clock() - start) / (step + 1)
                    logger.info(
                        "epoch {}, loss: {}, {}/step".format(epoch + 1, train_loss / num_batches, time_per_step))

            train_loss /= num_batches

            # validation loop
            output_path = f"{train_param.output_dir}/{epoch}_valid_result.txt"
            valid_f1_score = test_loop(idx_to_label, model, output_path, valid_generator)

            tf.summary.scalar('valid f1-score', valid_f1_score, step=(epoch * train_param.train_steps + step))
            writer.flush()

            # test loop
            if train_param.test_paths:
                output_path = f"{train_param.output_dir}/{epoch}_test_result.txt"
                test_f1_score = test_loop(idx_to_label, model, output_path, test_generator)
                tf.summary.scalar('test f1-score', test_f1_score, step=(epoch * train_param.train_steps + step))
                writer.flush()
                logger.info("epoch {}, loss: {}, valid f1-score: {}, test f1-score: {}".format(
                    epoch + 1, train_loss, valid_f1_score, test_f1_score))
            else:
                logger.info("epoch {}, loss: {}, valid f1-score: {}".format(
                    epoch + 1, train_loss, valid_f1_score))

            if do_nni:
                if train_param.test_paths:
                    nni_result = {'default': float(valid_f1_score),
                                  'test_f1_score': float(test_f1_score)}
                else:
                    nni_result = float(valid_f1_score)
                nni.report_intermediate_result(nni_result)

            if best_score < valid_f1_score:
                best_score = valid_f1_score
                best_epoch = epoch
                no_update = 0
                ckpt_manager.save()
                if 'test_f1_score' in locals():
                    last_test_score = test_f1_score
            else:
                no_update += 1

            # early stopping
            if 0 < train_param.early_stop_patience <= no_update:
                break

        # final report
        if do_nni:
            if train_param.test_paths:
                nni_result = {'default': float(best_score),
                              'test_f1_score': float(last_test_score)}
            else:
                nni_result = float(best_score)

            nni.report_final_result(nni_result)
        logger.info(f'best epoch: {best_epoch}')
        if train_param.test_paths:
            logger.info(f'best valid f1 score: {best_score}, with test f1 score: {last_test_score}')
        else:
            logger.info(f'best f1 score: {best_score}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--conf', type=str, default='conf/default.yaml')
    parser.add_argument('--do_nni', type=str2bool, nargs='?', default=False)
    args = parser.parse_args()
    train_param = load_param(args.conf)

    # nni
    if args.do_nni:
        tuned_params = nni.get_next_parameter()
        raw_params = train_param.dict()
        raw_params.update(tuned_params)
        raw_params['output_dir'] = os.environ['NNI_OUTPUT_DIR'] + "/checkpoint"
        train_param = TrainParam(**raw_params)
        train_param.overwrite = False

    # base setting
    if train_param.overwrite:
        shutil.rmtree(train_param.output_dir, ignore_errors=True)
    os.makedirs(train_param.output_dir, exist_ok=True)

    with open(os.path.join(train_param.output_dir, 'conf.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(train_param.dict(), f, allow_unicode=True, default_flow_style=False)
    main(train_param, args.do_nni)
