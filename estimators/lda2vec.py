import tensorflow as tf
import numpy as np

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
            initializer=tf.orthogonal_initializer(gain=scalar))
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

    document_context = tf.matmul(document_proportions,
                                 topic_embedding, name="document_context")

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
        loss_nce = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=labels,
                inputs=context,
                num_sampled=params["negative_samples"],
                num_classes=params["vocabulary_size"],
                num_true=1,
                sampled_values=None
            ))

    with tf.variable_scope("lda_loss"):
        batch_size = tf.cast(tf.shape(features["doc_id"])[0], dtype=tf.float32)
        loss_lda = batch_size / params["num_documents"] * dirichlet_likelihood(
            document_embedding, params["alpha"])

    with tf.variable_scope("total_loss"):
        loss = loss_nce + params["lambda"] * loss_lda

    train_op = tf.train.AdamOptimizer(learning_rate=params["learning_rate"]).minimize(
        loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar("loss_nce", loss_nce)
        tf.summary.scalar("loss_lda", loss_lda)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)

my_feature_columns = []

_VOCAB_SIZE = 26863

COLUMN_NAMES = ["target", "doc_id"]

lda2vec = tf.estimator.Estimator(
    model_fn = lda2vec_model_fn,
    model_dir="built_models/lda2vec",
    params={
        "learning_rate": 0.01,
        "embedding_size": 256,
        "num_topics": 20,
        "num_documents": 1193,
        "lambda": 200,
        "temperature": 1.0,
        "alpha": 0.7,
        "vocabulary_size": 26863,
        "negative_samples": 15
    }
)

lda2vec.train(
    lambda: train_input_fn("experiments/twenty_newsgroups/train_data.csv", 4096),
    steps=10000
)
