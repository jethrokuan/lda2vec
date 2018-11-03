import os
import json

import gin

class DataLoader(object):

    def __init__(self, data_path):
        file_freq = os.path.join(data_path, "freq.json")
        file_idx2token = os.path.join(data_path, "idx2token.json")
        file_token2idx = os.path.join(data_path, "token2idx.json")
        file_meta = os.path.join(data_path, "meta.json")
        file_train_path = os.path.join(data_path, "train.tfrecord")

        self.train_path = file_train_path

        with open(file_freq, "r") as fp:
            self.freq = json.load(fp)

        with open(file_idx2token, "r") as fp:
            self.idx2token = json.load(fp)

        with open(file_token2idx, "r") as fp:
            self.token2idx = json.load(fp)

        with open(file_meta, "r") as fp:
            self.meta = json.load(fp)
