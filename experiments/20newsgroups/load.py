from tensorflow.python.keras.preprocessing.sequence import skipgrams
import tensorflow as tf
from dataset_tools.preprocess import preprocess
from utils.dirs import create_dirs
from dataset_tools.utils import read_file
import json
import os

from collections import defaultdict

CORPUS = "data/twenty_newsgroups/train.txt"
texts = read_file(CORPUS)
SAVE_DIR = "data/twenty_newsgroups/"
create_dirs([SAVE_DIR])

_OOV_TOKEN = "<OOV>"
_OOV_TOKEN_ID = -1

dct, tokenized_docs = preprocess(texts, stem=False)
_OOV_TOKEN_ID = dct.token2id[_OOV_TOKEN]

frequency = defaultdict(int)

data = []

# data is an array containing examples
# TODO: data is now stored in memory, probably
# need to move this to a spark/beam job later on.

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tfrecord_path = os.path.join(SAVE_DIR, "train.tfrecord")

with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
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
                feature = {
                    "target": _int64_feature(pair[0]),
                    "context": _int64_feature(pair[1]),
                    "doc_id": _int64_feature(idx)
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())


total_count = sum(frequency.values())
normalized_frequency = {k: v / total_count for k, v in frequency.items()}

with open("{}/token2idx.json".format(SAVE_DIR), "w") as fp:
    json.dump(dct.token2id, fp)

id2token = {v: k for k, v in dct.token2id.items()}

with open("{}/idx2token.json".format(SAVE_DIR), "w") as fp:
    json.dump(id2token, fp)

with open("{}/freq.json".format(SAVE_DIR), "w") as fp:
    json.dump(normalized_frequency, fp)

with open("{}/meta.json".format(SAVE_DIR), "w") as fp:
    json.dump({
        "vocab_size": len(dct),
        "num_docs": len(tokenized_docs)
    }, fp)

with open("{}/word_embeddings.tsv".format(SAVE_DIR), "w") as fp:
    for _, v in dct.items():
        fp.write("{}\n".format(v))
