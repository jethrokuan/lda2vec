import pickle

import numpy as np
import json
import pandas as pd
from tensorflow.python.keras.preprocessing.sequence import skipgrams

from dataset_tools.preprocess import NlpPipeline
from dataset_tools.utils import read_file
from utils.dirs import create_dirs

CORPUS = "datasets/twenty_newsgroups.txt"
EXPERIMENT_DIR = "experiments/twenty_newsgroups/"
create_dirs([EXPERIMENT_DIR])

texts = read_file(CORPUS)
pipeline = NlpPipeline(texts, max_length=1000)

pipeline.tokenize()
pipeline.compact_documents()

data = []

for idx, document in enumerate(pipeline.compact_docs):
    pairs, _ = skipgrams(
        document,
        vocabulary_size=len(pipeline.vocab),
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
df.to_csv("{}/train_data.csv".format(EXPERIMENT_DIR), index=False, header=True)

np.save("{}/freqs".format(EXPERIMENT_DIR), pipeline.token_counts)
with open("{}/idx_to_word.pickle".format(EXPERIMENT_DIR), "wb") as fp:
    pickle.dump(pipeline.idx_to_word, fp)

with open("{}/word_to_idx.pickle".format(EXPERIMENT_DIR), "wb") as fp:
    pickle.dump(pipeline.word_to_idx, fp)

with open("{}/word_embedding_metadata.tsv".format(EXPERIMENT_DIR), "w") as fp:
    words = sorted(pipeline.idx_to_word.items())
    for _, word in words:
        fp.write("{}\n".format(word))

with open("{}/corpus_meta".format(EXPERIMENT_DIR), "w") as fp:
    json.dump({
        "vocab_size": len(pipeline.vocab),
        "doc_count": len(pipeline.compact_docs),
    }, fp)
