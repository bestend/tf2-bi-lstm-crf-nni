# -*- coding: utf-8-*-
from typing import Dict, Any

import numpy as np
import tensorflow as tf
import tensorflow_addons as tf_ad
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Conv1D, GlobalMaxPooling1D
from tensorflow.python.keras.layers import add

from feature_extractor import Vocabulary
from train_param import TrainParam


class BiLstmCrfModel(tf.keras.Model):
    def __init__(self, train_param: TrainParam, vocab: Vocabulary):
        super().__init__()
        self.train_param = train_param

        if train_param.use_elmo:
            # tf2 not support tf-hub version elmo
            '''
            elmo = hub.KerasLayer("https://tfhub.dev/google/elmo/3",
                                  trainable=True,
                                  signature="tokens",
                                  output_key=train_param.elmo_output)

            def elmo_embedding(x):
                tokens, sequence_len = x
                return elmo(
                    inputs={
                        'tokens': tf.squeeze(tf.cast(tokens, tf.string)),
                        'sequence_len': sequence_len
                    }
                )[train_param.elmo_output]

            self.embedding = Lambda(elmo_embedding, output_shape=(train_param.max_word, train_param.emb_dim),
                                    name='word_embedding')
            '''
        else:
            self.embedding = Embedding(len(vocab.token), train_param.emb_dim, name='word_embedding')

        self.pos_embedding = Embedding(len(vocab.pos), train_param.pos_emb_dim, name='pos_embedding')
        self.char_embedding = Embedding(len(vocab.char), train_param.char_emb_dim,
                                        input_length=(train_param.max_word, train_param.max_char,),
                                        name='char_embedding')
        if train_param.char_embedding_type == 'lstm':
            self.char_lstm = TimeDistributed(Bidirectional(LSTM(
                units=train_param.char_lstm_units,
                recurrent_dropout=train_param.char_lstm_recurrent_dropout,
                dropout=train_param.char_lstm_dropout,
            ), name='char_blstm'))
        else:
            self.char_cnn = TimeDistributed(
                Conv1D(train_param.char_cnn_filter_num, train_param.char_cnn_window_size, padding='same'),
                name="char_cnn")
            self.char_max_pool = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling")
            self.char_dropout = tf.keras.layers.Dropout(train_param.char_dropout)

        self.token_dropout = tf.keras.layers.Dropout(train_param.token_dropout)
        self.pos_dropout = tf.keras.layers.Dropout(train_param.pos_dropout)

        label_size = len(vocab.label)
        self.bilstm_layers = []
        for idx in range(train_param.lstm_layer_size):
            self.bilstm_layers.append(Bidirectional(LSTM(
                units=train_param.lstm_units,
                recurrent_dropout=train_param.lstm_recurrent_dropout,
                dropout=train_param.lstm_dropout,
                return_sequences=True
            ), name=f"blstm-{idx}"))
        self.dense = tf.keras.layers.Dense(label_size, name='logits')
        self.transition_params = tf.Variable(tf.random.uniform(shape=(label_size, label_size)))

    def load_embedding_weights(self, wv_path: str, vocab: Vocabulary):
        word_vectors = {}
        with open(wv_path) as f:
            for line in f.readlines()[1:]:
                line = line.split()
                word, vector = line[0], np.array(line[1:], dtype='float32')
                word_vectors[word] = vector

        self.embedding.build((None,))
        matrix = self.embedding.get_weights()[0]

        oov_count = 0
        for word, idx in vocab.token.items():
            target = word.lower() if self.train_param.do_lowercase else word

            if target in word_vectors:
                matrix[idx] = word_vectors[target]
            else:
                oov_count += 1

        print("wv vector oov rate: {}".format(oov_count / len(vocab.token)))
        self.embedding.set_weights([matrix])

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs: Dict[str, Any], labels=None, training=None):
        # token
        if self.train_param.use_elmo:
            # tf2 not support tf-hub version elmo
            '''
            token_embedding = self.embedding([inputs['raw_token'], inputs['sequence_lengths']])
            '''
            token_embedding = inputs['token']
        else:
            token_embedding = self.embedding(inputs['token'])
        token_embedding = self.token_dropout(token_embedding, training=training)
        # pos
        pos_embedding = self.pos_embedding(inputs['pos'])
        pos_embedding = self.pos_dropout(pos_embedding, training=training)
        # char
        char_embedding = self.char_embedding(inputs['char'])
        if self.train_param.char_embedding_type == 'lstm':
            char_embedding = self.char_lstm(char_embedding, training=training)
        else:
            char_embedding = self.char_cnn(char_embedding)
            char_embedding = self.char_max_pool(char_embedding)
            char_embedding = self.char_dropout(char_embedding, training=training)

        feature = tf.concat([token_embedding, char_embedding, pos_embedding], axis=-1)

        immediate_feature = []
        for layer in self.bilstm_layers:
            immediate_feature.append(layer(feature, training=training))
            feature = immediate_feature[-1]
        if len(self.bilstm_layers) > 1:
            feature = add(immediate_feature)  # residual
        logits = self.dense(feature)

        if labels is not None:
            log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(logits,
                                                                                   labels,
                                                                                   inputs['sequence_lengths'],
                                                                                   transition_params=self.transition_params)
            return logits, log_likelihood
        else:
            return logits
