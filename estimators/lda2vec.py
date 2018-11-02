import uuid
import tensorflow as tf
import numpy as np
import itertools

import os
import json

from argparse import ArgumentParser
from dataset_tools.data_loader import DataLoader

from sklearn.preprocessing import normalize

parser = ArgumentParser()

tf.logging.set_verbosity(tf.logging.INFO)

dataloader = DataLoader("data/twenty_newsgroups/")

def build_input_fn(tfrecord_path, batch_size, cache=True):
    def input_fn():
        def parse(serialized):
            features = {
                'doc_id': tf.FixedLenFeature([], tf.int64),
                'target': tf.FixedLenFeature([], tf.int64),
                'context': tf.FixedLenFeature([], tf.int64)
            }

            parsed_example = tf.parse_single_example(
                serialized=serialized, features=features)

            input = {
                'doc_id': tf.cast(parsed_example["doc_id"], tf.int32),
                'target': tf.cast(parsed_example["target"], tf.int32)
            }

            label = tf.cast(parsed_example["context"], tf.int32)

            return input, label


        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(parse)
        dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1000))
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(2)
        if cache:
            dataset = dataset.cache()
        batch = dataset.make_one_shot_iterator().get_next()
        return batch
    return input_fn

def lda2vec_model_fn(features, labels, mode, params):
    """LDA2vec model."""

    def dirichlet_likelihood(weights, alpha):
        log_proportions = tf.nn.log_softmax(weights)
        loss = (alpha - 1.0) * log_proportions
        return tf.reduce_sum(loss)

    with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
        scalar = 1 / np.sqrt(params["num_documents"] + params["num_topics"])
        word_embedding = tf.get_variable(
            "word_embedding",
            shape=[params["vocabulary_size"],
                   params["embedding_size"]],
            dtype=tf.float32,
            initializer=tf.initializers.random_normal())
        topic_embedding = tf.get_variable(
            "topic_embedding",
            shape=[params["num_topics"], params["embedding_size"]],
            dtype=tf.float32,
            initializer=tf.orthogonal_initializer(gain=scalar)
        )
        document_embedding = tf.get_variable(
            "document_embedding",
            shape=[params["num_documents"], params["num_topics"]],
            dtype=tf.float32,
            initializer=tf.initializers.random_normal(
                mean=0.0,
                stddev= 50 * scalar))

    word_context = tf.nn.embedding_lookup(
        word_embedding, features["target"], name="word_context")

    document_proportions = tf.nn.embedding_lookup(
        document_embedding,
        features["doc_id"],
        name="document_proportions"
    )

    document_softmax = tf.nn.softmax(
        document_proportions / params["temperature"], name="document_softmax")

    document_context = tf.matmul(document_softmax,
                                 topic_embedding, name="document_context")

    # word_context = tf.nn.dropout(word_context, keep_prob=params["dropout_ratio"])
    # document_context = tf.nn.dropout(document_context, keep_prob=params["dropout_ratio"])

    # word_context = tf.nn.l2_normalize(word_context, name="normalize_word")
    # word_context = tf.nn.l2_normalize(document_context, name="normalize_document")

    contexts_to_add = [word_context, document_context]

    context = tf.add_n(contexts_to_add, name="context_vector")

    with tf.variable_scope("nce_loss"):
        nce_weights = tf.Variable(
            tf.truncated_normal(
                [params["vocabulary_size"],
                 params["embedding_size"]],
                stddev=tf.sqrt(1/params["embedding_size"])
            ),
            name="nce_weights")
        nce_biases = tf.Variable(tf.zeros(params["vocabulary_size"]), name="nce_biases")
        labels = tf.reshape(labels, [tf.shape(labels)[0], 1])
        sampler = tf.nn.learned_unigram_candidate_sampler(
            true_classes=tf.cast(labels, tf.int64),
            num_true=1,
            num_sampled=params["negative_samples"],
            unique=True,
            range_max=params["vocabulary_size"],
            name="sampler"
        )
        loss_nce = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=labels,
                inputs=context,
                num_sampled=params["negative_samples"],
                num_classes=params["vocabulary_size"],
                num_true=1,
                sampled_values=sampler
            ))

    with tf.variable_scope("lda_loss"):
        # regularizer = tf.contrib.layers.l1_regularizer(scale=1.0)
        # loss_lda = regularizer(document_proportions)
        batch_size = tf.cast(tf.shape(features["doc_id"])[0], dtype=tf.float32)
        loss_lda = batch_size / params["num_documents"] * dirichlet_likelihood(
            document_proportions, params["alpha"])

    loss = loss_nce + params["lambda"] * loss_lda

    train_op = tf.contrib.opt.LazyAdamOptimizer(learning_rate=params["learning_rate"]).minimize(
        loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar("loss_nce", loss_nce)
        tf.summary.scalar("loss_lda", loss_lda)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)


my_feature_columns = []

COLUMN_NAMES = ["target", "doc_id"]

params = {
    "learning_rate": 0.001,
    "embedding_size": 256,
    "num_topics": 15,
    "num_documents": dataloader.meta["num_docs"],
    "lambda": 200,
    "temperature": 1.0,
    "alpha": 0.7,
    "vocabulary_size": dataloader.meta["vocab_size"],
    "negative_samples": 15,
    "dropout_ratio": 0.5
}

model_dir = "built_models/test_{}".format(uuid.uuid1())

lda2vec = tf.estimator.Estimator(
    model_fn = lda2vec_model_fn,
    model_dir=model_dir,
    params=params
)

early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
    lda2vec,
    metric_name="loss",
    max_steps_without_decrease=1000,
    min_steps=1000
)

log_tensors = tf.train.LoggingTensorHook(
    tensors=["embeddings/topic_embedding", "embeddings/word_embedding", # "document_context", "word_context"
    ],
    every_n_iter=1000,
)

profiler_hook = tf.train.ProfilerHook(
    save_steps=100, show_dataflow=True, show_memory=True, output_dir=model_dir)

lda2vec.train(
    input_fn=build_input_fn(dataloader.train_path, 64),
    max_steps=1000,
    hooks = [profiler_hook]
)

def get_topics(estimator):
    """Gets the topics for a given estimator.

    Args:
       estimator: trained lda2vec estimator.

    Returns:
       None. Prints the topics for the trained model.
    """
    topic_embedding = estimator.get_variable_value("embeddings/topic_embedding:0")
    word_embedding = estimator.get_variable_value("embeddings/word_embedding:0")

    topic_embedding = normalize(topic_embedding, norm='l2')
    word_embedding = normalize(word_embedding, norm='l2')

    cosine_sim = np.matmul(topic_embedding, np.transpose(word_embedding))

    for idx, topic in enumerate(cosine_sim):
        top_k = topic.argsort()[::-1][:10]
        nearest_words = list(map(dataloader.idx2token.get, map(str, top_k)))
        print("Topic {}: {}".format(
            idx,
            nearest_words
        ))

get_topics(lda2vec)
