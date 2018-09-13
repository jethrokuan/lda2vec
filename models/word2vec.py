"""Basic Word2vec model."""

from models.model import BaseModel
import tensorflow as tf
import math


class Word2Vec(BaseModel):
    def __init__(self, config, hparams):
        super(Word2Vec, self).__init__(config)
        self.hparams = hparams
        self.build_graph()
        self.init_saver()

    def build_graph(self):
        with tf.name_scope("inputs"):
            self.train_inputs = tf.placeholder(tf.int32, shape=[None])
            self.train_labels = tf.placeholder(tf.int32, shape=[None])

        with tf.device("/cpu:0"):
            with tf.name_scope("embeddings"):
                embeddings = tf.Variable(
                    tf.random_uniform([
                        self.hparams.vocabulary_size,
                        self.hparams.embedding_size
                    ], -1.0, 1.0))
                self.embed = tf.nn.embedding_lookup(embeddings,
                                                    self.train_inputs)

        with tf.name_scope("weights"):
            nce_weights = tf.Variable(
                tf.truncated_normal(
                    [
                        self.hparams.vocabulary_size,
                        self.hparams.embedding_size
                    ],
                    stddev=1.0 / math.sqrt(self.hparams.embedding_size)))
        with tf.name_scope("biases"):
            nce_biases = tf.Variable(
                tf.zeros([self.hparams.vocabulary_size]))

        with tf.name_scope("loss"):
            train_labels_reshaped = tf.reshape(
                self.train_labels, [tf.shape(self.train_labels)[0], 1])
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels_reshaped,
                    inputs=self.embed,
                    num_sampled=self.hparams.negative_samples,
                    num_classes=self.hparams.vocabulary_size,
                    name="nce_loss"))
            self.train_step = tf.train.AdamOptimizer(
                self.hparams.learning_rate).minimize(
                    self.loss, global_step=self.global_step_tensor)
