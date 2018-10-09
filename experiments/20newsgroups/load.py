from tensorflow.python.keras.preprocessing.sequence import skipgrams
import tensorflow as tf
from dataset_tools.preprocess import preprocess
from utils.dirs import create_dirs
from dataset_tools.utils import read_file
import json
import pandas as pd

from collections import defaultdict

CORPUS = "datasets/twenty_newsgroups/train.txt"
texts = read_file(CORPUS)
SAVE_DIR = "datasets/twenty_newsgroups/"
create_dirs([SAVE_DIR])

_OOV_TOKEN = "<OOV>"
_OOV_TOKEN_ID = -1

dct, tokenized_docs = preprocess(texts, stem=True)

frequency = defaultdict(int)

data = []

for idx, doc in enumerate(tokenized_docs):
    id_doc = dct.doc2idx(doc, _OOV_TOKEN_ID)
    for token_id in id_doc:
        frequency[token_id] += 1
    pairs, _ = skipgrams(
        id_doc,
        vocabulary_size=len(dct),
        window_size=5,
        shuffle=True,
        negative_samples=0)

    if len(pairs) > 2:
        for pair in pairs:
            ex = {}
            ex["target"], ex["context"] = pair
            ex["doc_id"] = idx
            data.append(ex)

df = pd.DataFrame(data)
df.to_csv("{}/train.csv".format(SAVE_DIR), index=False, header=True)

total_count = sum(frequency.values())
normalized_frequency = {k: v / total_count for k, v in frequency.items()}

token_to_idx = dct.token2id
token_to_idx[_OOV_TOKEN] = _OOV_TOKEN_ID

with open("{}/token_to_idx.json".format(SAVE_DIR), "w") as fp:
    json.dump(token_to_idx, fp)

with open("{}/freq.json".format(SAVE_DIR), "w") as fp:
    json.dump(normalized_frequency, fp)

with open("{}/meta.json".format(SAVE_DIR), "w") as fp:
    json.dump({
        "vocab": len(token_to_idx),
        "num_docs": len(tokenized_docs)
    }, fp)
