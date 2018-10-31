import uuid
import tensorflow as tf
import numpy as np
import itertools

import os
import json

from argparse import ArgumentParser
from dataset_tools.data_loader import DataLoader

parser = ArgumentParser()

tf.logging.set_verbosity(tf.logging.INFO)

dataloader = DataLoader("data/twenty_newsgroups/no_oov")

def train_input_fn(f, batch_size):
    dataset = tf.contrib.data.make_csv_dataset(
        f,
        batch_size=batch_size,
        label_name="context"
    )
    dataset = dataset.repeat()
    batch = dataset.make_one_shot_iterator().get_next()
    return batch

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
            # initializer=tf.orthogonal_initializer(gain=scalar)
            initializer=tf.orthogonal_initializer(gain=5)
        )
        document_embedding = tf.get_variable(
            "document_embedding",
            shape=[params["num_documents"], params["num_topics"]],
            dtype=tf.float32,
            initializer=tf.initializers.random_normal(
                mean=0.0,
                stddev= 50 * scalar))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {}
        topic_embedding = tf.nn.l2_normalize(topic_embedding, 1)
        word_embedding = tf.nn.l2_normalize(word_embedding, 1)
        with tf.variable_scope("k_closest"):
            indices = np.arange(params["num_topics"])
            topic = tf.nn.embedding_lookup(topic_embedding, indices)
            cosine_sim = tf.matmul(topic, tf.transpose(word_embedding, [1, 0]))
            sim, sim_idxs = tf.nn.top_k(cosine_sim, k=10)
            predictions["top_k"] = sim
            predictions["sim_idxs"] = sim_idxs
            return tf.estimator.EstimatorSpec(
                mode, predictions=predictions
            )

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
        batch_size = tf.cast(tf.shape(features["doc_id"])[0], dtype=tf.float32)
        loss_lda = batch_size / params["num_documents"] * dirichlet_likelihood(
            document_embedding, params["alpha"])

    loss = loss_nce + params["lambda"] * loss_lda

    train_op = tf.train.AdagradOptimizer(learning_rate=params["learning_rate"]).minimize(
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
    "alpha": 0.5,
    "vocabulary_size": dataloader.meta["vocab_size"],
    "negative_samples": 15,
    "dropout_ratio": 0.5
}

lda2vec = tf.estimator.Estimator(
    model_fn = lda2vec_model_fn,
    model_dir="built_models/test_{}".format(uuid.uuid1()),
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

lda2vec.train(
    input_fn=lambda: train_input_fn(dataloader.train, 64),
    max_steps=500000,
    # hooks = [log_tensors]
)

# topics = lda2vec.get_variable_value("embeddings/topic_embedding:0")
# words = lda2vec.get_variable_value("embeddings/word_embedding:0")
# documents = lda2vec.get_variable_value("embeddings/document_embedding:0")
# proportions = lda2vec.get_variable_value("document_softmax:0")

# print(topics)
# print(words)
# print(documents)
# print(proportions)

predictions = lda2vec.predict(
    input_fn=lambda: ({}, []))
predictions = itertools.islice(predictions, params["num_topics"])

for idx, pred in enumerate(predictions):
    print("Topic {}: {}".format(
        idx,
        list(map(dataloader.idx2token.get, map(str, pred["sim_idxs"])))
    ))
