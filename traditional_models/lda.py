"""Train LDA on a dataset.

Usage:
    python traditional_models/lda.py \
      --data_path datasets/twenty_newsgroups.txt
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser

from time import time

import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dataset_tools.utils import read_file

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib

parser = ArgumentParser()

parser.add_argument(
    "--data_path", help="Path to newline-delimited text file", required=True)
parser.add_argument(
    "--save_path", help="Path to save lda model.", required=True)

args, _ = parser.parse_known_args()

if not os.path.exists(args.data_path):
    raise Exception("Invalid data path {}".format(args.data_path))

data = read_file(args.data_path)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

t0 = time()

n_features = 1000
n_components = 10
n_top_words = 10


count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
tf = count_vectorizer.fit_transform(data)

logging.info("count vectorizer completed in {} seconds".format(time() - t0))

logging.info("Fitting LDA model with tf features, n_features={}".format(n_features))
t0 = time()

lda = LatentDirichletAllocation(n_components=n_components,
                                max_iter=100,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)
logging.info("done in %0.3fs." % (time() - t0))

joblib.dump(lda, args.save_path)

print("\nTopics in Topic Model:")
tf_feature_names = count_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
