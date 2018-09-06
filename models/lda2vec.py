"""Lda2vec model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.model import BaseModel

from models.lda2vec.embed_mixture import EmbedMixture


class Lda2Vec(BaseModel):
    def __init__(self, config, hparams):
        super(BaseModel, self).__init__(config)
        self.build_graph()
        self.hparams = hparams

    def build_graph(self):
        """
        pivot_idxs: pivot words (int)
        target_idxs: context words (int)
        doc_ids: docs at pivot (int)
        """
        pivot_idxs = tf.placeholder(tf.int32, shape=[None], name="pivot_idxs")
        target_idxs = tf.placeholder(
            tf.int64, shape=[None], name="target_idxs")
        doc_ids = tf.placeholder(tf.int32, shape=[None], name="doc_ids")

        word_embedding = tf.Variable(
            tf.random_uniform(
                [self.hparams.vocab_size, self.hparams.embedding_size], -1.0,
                1.0),
            name="word_embedding")

        doc_embedding = tf.Variable(tf.random_normal([n_documents, n_topics], mean=0, stddev=50 * scalar),
                                    name=self.name+ "_" +"doc_embedding")

        with tf.variable_scope("nce_loss"):
            nce_weights = tf.Variable(tf.truncated_normal(
                [self.hparams.vocab_size, self.hparams.embedding_size],
                stddev=tf.sqrt(1 / self.hparams.embedding_size)),
                                      name="nce_weights")
            nce_biases = tf.Variable(tf.zeros([vocab_size]), name="nce_biases")
            target_labels = tf.reshape(target_idxs, [tf.shape(target_idxs)[0], 1])
            nce_loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=target_labels,
                               inputs=context,
                               num_sampled=self.hparams.negative_sampling_size,
                               num_classes=self.hparams.vocab_size,
                               num_true=1,
                               sampled_values=None))
