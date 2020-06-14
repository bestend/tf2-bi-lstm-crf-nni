# -*- coding: utf-8-*-
import os
from typing import List, Dict, Any

from pydantic import BaseModel, validator, root_validator


class TrainParam(BaseModel):
    train_paths: List[str]
    valid_paths: List[str]
    test_paths: List[str] = []
    vocab_path: str
    output_dir: str
    overwrite: bool = True
    pretrained_embedding_path: str = ''
    do_lowercase: bool = True

    rand_seed: int = 777
    batch_size: int = 128
    optimizer: str = 'adam'
    learning_rate: float = 2e-3
    lr_decay_rate: float = 1e-5
    optimizer_param: Dict[str, Any] = {}
    epochs: int = 10
    early_stop_patience: int = 3
    train_steps: int = 1000

    # elmo (https://tfhub.dev/google/elmo/3)
    use_elmo: bool = False
    elmo_output: str = 'elmo'
    emb_dim: int = 200  # elmo 를 사용하는 경우 현재 입력된 값이 무시됨
    char_emb_dim: int = 50
    pos_emb_dim: int = 30
    max_word: int = 100
    max_char: int = 10

    token_dropout: float = 0.3
    pos_dropout: float = 0.0
    char_dropout: float = 0.3

    lstm_units: int = 400
    lstm_recurrent_dropout: float = 0.2
    lstm_dropout: float = 0.2
    lstm_layer_size: int = 1

    char_embedding_type: str = 'cnn'

    char_lstm_units: int = 40
    char_lstm_recurrent_dropout: float = 0.2
    char_lstm_dropout: float = 0.2

    char_cnn_window_size: int = 3
    char_cnn_filter_num: int = 30

    @validator('elmo_output')
    def validate_elmo_output(cls, elmo_output):
        if elmo_output not in ['word_emb', 'lstm_outputs1', 'lstm_outputs2', 'elmo']:
            raise ValueError(f'elmo_output은 다음 형태만 지원합니다. -> (word_emb, lstm_outputs1, lstm_outputs2, elmo)')
        return elmo_output

    @validator('char_embedding_type')
    def validate_char_embedding_type(cls, char_embedding_type):
        if char_embedding_type not in ['lstm', 'cnn']:
            raise ValueError(f'char_embedding_type 다음 형태만 지원합니다. -> (lstm, cnn)')
        return char_embedding_type

    @validator('train_paths', 'valid_paths', 'vocab_path', each_item=True)
    def validate_path(cls, path):
        if not os.path.exists(path):
            raise ValueError(f'path: 파일이 존재하지 않습니다. {path}')
        return path

    @root_validator
    def elmo_embed_dim(cls, values):
        use_elmo, elmo_output = values.get('use_elmo'), values.get('elmo_output')
        if use_elmo:
            if elmo_output == 'word_emb':
                values['embed_dim'] = 512
            else:
                values['embed_dim'] = 1024
        return values

    class Config:
        extra = 'forbid'
