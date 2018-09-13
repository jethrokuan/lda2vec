"""Lda2vec model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from models.model import BaseModel


class Lda2Vec(BaseModel):
    def __init__(self, config, hparams):
        """Constructor for Lda2Vec model.

        Args:
            config: Settings for model training and evaluation.
            hparams: a tf HParams object.
        """
        super(Lda2Vec, self).__init__(config)
        self.config = config
        self.hparams = hparams
        self.build_graph()
        self.init_saver()

    def dirichlet_likelihood(self, weights, alpha):
        log_proportions = tf.nn.log_softmax(weights)
        loss = (alpha - 1.0) * log_proportions
        return tf.reduce_sum(loss)

    def _embed_mixture(self,
                       num_docs,
                       num_topics,
                       embedding_size,
                       name):
        scalar = 1 / np.sqrt(num_docs + num_topics)
        document_embedding = tf.Variable(
            tf.random_normal(
                [num_docs, num_topics], mean=0, stddev=50 * scalar),
            name=name)
        with tf.name_scope("{}_Topics".format(name)):
            topic_embedding = tf.get_variable(
                "{}_topic_embedding".format(name),
                shape=[self.hparams.num_topics, embedding_size],
                dtype=tf.float32,
                initializer=tf.orthogonal_initializer(gain=scalar))

        return document_embedding, topic_embedding

    def build_graph(self):
        """
        pivot_idxs: pivot words (int)
        target_idxs: context words (int)
        doc_ids: docs at pivot (int)
        """
        with tf.name_scope("inputs"):
            self.train_inputs = tf.placeholder(tf.int32, shape=[None], name="target_idxs")
            self.train_labels = tf.placeholder(
                tf.int64, shape=[None], name="context_idxs")
            self.doc_ids = tf.placeholder(tf.int32, shape=[None], name="doc_ids")

        word_embeddings = tf.Variable(
            tf.random_uniform(
                [self.hparams.vocabulary_size, self.hparams.embedding_size], -1.0,
                1.0),
            name="word_embedding")

        word_context = tf.nn.embedding_lookup(
            word_embeddings, self.train_inputs, name="word_context")

        document_embedding, topic_embedding = self._embed_mixture(
            self.config["num_documents"], self.hparams.num_topics,
            self.hparams.embedding_size, name="document")

        document_proportions = tf.nn.embedding_lookup(
            document_embedding,
            self.doc_ids,
            name="{}_doc_proportions".format("document"))

        document_proportions = tf.nn.softmax(
            document_proportions / self.hparams.temperature, name="document_softmax")

        document_context = tf.matmul(document_proportions,
                                     topic_embedding, name="document_context")

        contexts_to_add = [word_context, document_context]

        context = tf.add_n(contexts_to_add, name="context_vector")

        with tf.variable_scope("nce_loss"):
            nce_weights = tf.Variable(
                tf.truncated_normal(
                    [
                        self.hparams.vocabulary_size,
                        self.hparams.embedding_size
                    ],
                    stddev=tf.sqrt(1 / self.hparams.embedding_size)),
                name="nce_weights")
            nce_biases = tf.Variable(tf.zeros([self.hparams.vocabulary_size]), name="nce_biases")
            train_labels = tf.reshape(self.train_labels,
                                       [tf.shape(self.train_labels)[0], 1])
            loss_nce = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs=context,
                    num_sampled=self.hparams.negative_samples,
                    num_classes=self.hparams.vocabulary_size,
                    num_true=1,
                    sampled_values=None))

        with tf.variable_scope("lda_loss"):
            fraction = tf.Variable(
                1, trainable=False, dtype=tf.float32, name="fraction")
            loss_lda = self.hparams.alpha * fraction * self.dirichlet_likelihood(
                document_embedding, self.hparams.alpha)

        with tf.variable_scope("total_loss"):
            self.loss = loss_nce + loss_lda

        self.train_step = tf.train.AdamOptimizer(
            self.hparams.learning_rate).minimize(
                self.loss, global_step=self.global_step_tensor)
