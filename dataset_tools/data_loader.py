import os
import json

class DataLoader(object):

    def __init__(self, data_path):
        file_freq = os.path.join(data_path, "freq.json")
        file_idx2token = os.path.join(data_path, "idx2token.json")
        file_token2idx = os.path.join(data_path, "token2idx.json")
        file_meta = os.path.join(data_path, "meta.json")
        file_train_csv = os.path.join(data_path, "train.csv")

        if os.path.exists(file_train_csv):
            self.train = file_train_csv

        with open(file_freq, "r") as fp:
            self.freq = json.load(fp)

        with open(file_idx2token, "r") as fp:
            self.idx2token = json.load(fp)

        with open(file_token2idx, "r") as fp:
            self.token2idx = json.load(fp)

        with open(file_meta, "r") as fp:
            self.meta = json.load(fp)
