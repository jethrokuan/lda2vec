CORPUS="datasets/twenty_newsgroups.txt"
EXPERIMENT_DIR="experiments/twenty_newsgroups/"

from keras.preprocessing.sequence import skipgrams

from dataset_tools.utils import read_file
from dataset_tools.preprocess import NlpPipeline

import numpy as np
import pickle
import pandas as pd

texts = read_file(CORPUS)
pipeline = NlpPipeline(texts, max_length=10000)

pipeline.tokenize()
pipeline.compact_documents()
pipeline.initialize_embedding_matrix()

data = []

for idx, document in enumerate(pipeline.compact_docs):
    pairs, _ = skipgrams(document,
                         vocabulary_size=len(pipeline.vocab),
                         window_size=5,
                         shuffle=True,
                         negative_samples=0)

    if len(pairs) > 2:
        for pair in pairs:
            temp_data = pair
            temp_data.append(idx)
            data.append(temp_data)

df = pd.DataFrame(data)
df.to_csv("{}/train_data.tsv".format(EXPERIMENT_DIR),
          sep="\t",
          index=False,
          header=None,
          mode="a")

np.save("{}/embed_matrix".format(EXPERIMENT_DIR),
        pipeline.embed_matrix)
np.save("{}/freqs".format(EXPERIMENT_DIR),
        pipeline.token_counts)
with open("{}/idx_to_word.pickle".format(EXPERIMENT_DIR), "wb") as fp:
    pickle.dump(pipeline.idx_to_word, fp)

with open("{}/word_to_idx.pickle".format(EXPERIMENT_DIR), "wb") as fp:
    pickle.dump(pipeline.word_to_idx, fp)
