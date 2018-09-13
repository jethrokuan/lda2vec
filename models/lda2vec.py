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

    def dirichlet_likelihood(weights, alpha=None):
        n_topics = weights.get_shape()[1].value
        if alpha is None:
            alpha = 1.0 / n_topics
        log_proportions = tf.nn.log_softmax(weights)
        loss = (alpha - 1.0) * log_proportions
        return tf.reduce_sum(loss)

    def _embed_mixture(num_docs,
                       num_topics,
                       embedding_size,
                       temperature=1.0,
                       name=""):
        scalar = 1 / np.sqrt(num_docs + num_topics)
        document_embedding = tf.Variable(
            tf.random_normal(
                [num_docs, num_topics], mean=0, stddev=50 * scalar),
            name=name)
        with tf.name_scope("{}_Topics".format(name)):
            topic_embedding = tf.get_variable(
                "{}_topic_embedding",
                shape=[n_topics, embedding_size],
                dtype=tf.float32,
                initializer=tf.orthogonal_initializer(gain=scalar))

        return document_embedding, topic_embedding

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

        word_context = tf.nn.embedding_lookup(
            word_embedding, pivot_idxs, name="word_context")

        document_embedding, topic_embedding = self._embed_mixture(
            self.config["num_documents"], self.hparams["num_topics"],
            self.hparams["embedding_size"])

        document_proportions = tf.nn.embedding_lookup(
            document_embedding,
            doc_ids,
            name="{}_doc_proportions".format(name))
        softmaxed_document_proportions = tf.nn.softmax(
            document_proportions / temperature)

        document_context = tf.matmul(softmaxed_document_proportions,
                                     topic_embedding, "document_context")

        contexts_to_add = [word_context, document_context]

        context = tf.add_n(contexts_to_add, name="context_vector")

        with tf.variable_scope("nce_loss"):
            nce_weights = tf.Variable(
                tf.truncated_normal(
                    [
                        self.hparams["vocab_size"],
                        self.hparams["embedding_size"]
                    ],
                    stddev=tf.sqrt(1 / self.hparams["embedding_size"])),
                name="nce_weights")
            nce_biases = tf.Variable(tf.zeros([vocab_size]), name="nce_biases")
            target_labels = tf.reshape(target_idxs,
                                       [tf.shape(target_idxs)[0], 1])
            loss_nce = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=target_labels,
                    inputs=context,
                    num_sampled=self.hparams.negative_sampling_size,
                    num_classes=self.hparams.vocab_size,
                    num_true=1,
                    sampled_values=None))

        with tf.variable_scope("lda_loss"):
            fraction = tf.Variable(
                1, trainable=False, dtype=tf.float32, name="fraction")
            loss_lda = self.hparams["alpha"] * fraction * self.dirichlet_likelihood(
                document_embedding)

        self.loss = nce_loss + loss_lda
        self.train_step = tf.train.AdamOptimizer(
            self.hparams["learning_rate"]).minimize(
                self.loss, global_step=self.global_step_tensor)
