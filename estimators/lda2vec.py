import tensorflow as tf
import numpy as np
import gin.tf.external_configurables
from nltk.corpus import stopwords

from argparse import ArgumentParser
from dataset_tools.data_loader import DataLoader
from dataset_tools.embeddings import load_embeddings

from sklearn.preprocessing import normalize

tf.logging.set_verbosity(tf.logging.INFO)

parser = ArgumentParser()

parser.add_argument("--config", help="Path to gin config file.", required=True)

args = parser.parse_args()


@gin.configurable("input", blacklist=["tfrecord_path"])
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
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=1000))
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(2)
        if cache:
            dataset = dataset.cache()
        batch = dataset.make_one_shot_iterator().get_next()
        return batch

    return input_fn


@gin.configurable("model")
def build_model_fn(learning_rate, num_documents, num_topics,
                   vocabulary_size, embedding_size, alpha,
                   negative_samples, lda_loss_weight, temperature,
                   dropout_ratio, optimizer, switch_loss_step,
                   idx2token, pretrained_embeddings=None):

    word_embedding_matrix = np.random.uniform(-1, 1, size=(vocabulary_size, embedding_size)).astype("float32")
    if pretrained_embeddings:
        embeddings = load_embeddings(pretrained_embeddings)
        count = 0
        for i, w in idx2token.items():
            v = embeddings.get(w)
            if v is not None and int(i) < vocabulary_size:
                word_embedding_matrix[int(i)] = v
                count += 1
        tf.logging.info("Preloaded {} of {} in vocab.".format(count, len(idx2token)))

    def lda2vec_model_fn(features, labels, mode, params):
        """LDA2vec model."""

        def dirichlet_likelihood(weights, alpha):
            log_proportions = tf.nn.log_softmax(weights)
            loss = (alpha - 1.0) * log_proportions
            return tf.reduce_sum(loss)

        with tf.device("/cpu:0"):
            with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
                scalar = 1 / np.sqrt(num_documents + num_topics)
                word_embedding = tf.get_variable(
                    "word_embedding",
                    dtype=tf.float32,
                    initializer=word_embedding_matrix)
                document_embedding = tf.get_variable(
                    "document_embedding",
                    shape=[num_documents, num_topics],
                    dtype=tf.float32,
                    initializer=tf.initializers.random_normal(
                        mean=0.0, stddev=50 * scalar))

        with tf.device("/cpu:0"):
            word_context = tf.nn.embedding_lookup(
                word_embedding, features["target"], name="word_context")

            document_proportions = tf.nn.embedding_lookup(
                document_embedding,
                features["doc_id"],
                name="document_proportions")

        document_softmax = tf.nn.softmax(
            document_proportions / temperature, name="document_softmax")

        with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
            topic_embedding = tf.get_variable(
                    "topic_embedding",
                    shape=[num_topics, embedding_size],
                    dtype=tf.float32,
                    initializer=tf.orthogonal_initializer(gain=scalar))

        document_context = tf.matmul(
            document_softmax, topic_embedding, name="document_context")

        word_context = tf.nn.dropout(word_context, keep_prob=dropout_ratio)
        document_context = tf.nn.dropout(
            document_context, keep_prob=dropout_ratio)

        contexts_to_add = [word_context, document_context]

        context = tf.add_n(contexts_to_add, name="context_vector")

        with tf.variable_scope("nce_loss"):
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=tf.sqrt(1 / embedding_size)),
                name="nce_weights")
            nce_biases = tf.Variable(
                tf.zeros(vocabulary_size), name="nce_biases")
            labels = tf.reshape(labels, [tf.shape(labels)[0], 1])
            sampler = tf.nn.learned_unigram_candidate_sampler(
                true_classes=tf.cast(labels, tf.int64),
                num_true=1,
                num_sampled=negative_samples,
                unique=True,
                range_max=vocabulary_size,
                name="sampler")
            loss_nce = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=labels,
                    inputs=context,
                    num_sampled=negative_samples,
                    num_classes=vocabulary_size,
                    num_true=1,
                    sampled_values=sampler))

        with tf.variable_scope("lda_loss"):
            batch_size = tf.cast(
                tf.shape(features["doc_id"])[0], dtype=tf.float32)
            loss_lda = batch_size / num_documents * dirichlet_likelihood(
                document_proportions, alpha)

        global_step = tf.train.get_global_step()

        loss = tf.cond(
            global_step < switch_loss_step,
            lambda: loss_nce,
            lambda: loss_nce + lda_loss_weight * loss_lda,
        )

        train_op = optimizer(learning_rate=learning_rate).minimize(
            loss, global_step=tf.train.get_global_step())

        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar("loss_nce", loss_nce)
            tf.summary.scalar("loss_lda", loss_lda)
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)

    return lda2vec_model_fn


@gin.configurable
def train(data_path, model_dir, max_steps, profile=False):
    dataloader = DataLoader(data_path)
    model_fn = build_model_fn(
        num_documents=dataloader.meta["num_docs"],
        vocabulary_size=dataloader.meta["vocab_size"],
        idx2token=dataloader.idx2token)
    input_fn = build_input_fn(tfrecord_path=dataloader.train_path)
    lda2vec = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)
    hooks = []
    if profile:
        profiler_hook = tf.train.ProfilerHook(
            save_steps=10000,
            show_dataflow=True,
            show_memory=True,
            output_dir=model_dir)
        hooks.append(profiler_hook)

    lda2vec.train(input_fn=input_fn, max_steps=max_steps, hooks=hooks)

    get_topics(lda2vec, dataloader.idx2token)

    return lda2vec


def get_topics(estimator, idx2token):
    """Gets the topics for a given estimator.

    Args:
       estimator: trained lda2vec estimator.
       idx2token: idx2token mapping

    Returns:
       None. Prints the topics for the trained model.
    """
    topic_embedding = estimator.get_variable_value(
        "embeddings/topic_embedding:0")
    word_embedding = estimator.get_variable_value(
        "embeddings/word_embedding:0")

    topic_embedding = normalize(topic_embedding, norm='l2')
    word_embedding = normalize(word_embedding, norm='l2')

    cosine_sim = np.matmul(topic_embedding, np.transpose(word_embedding))

    for idx, topic in enumerate(cosine_sim):
        top_k = topic.argsort()[::-1][:30]
        nearest_words = list(map(idx2token.get, map(str, top_k)))
        # Remove stopwords
        nearest_words = [word for word in nearest_words if word not in set(stopwords.words("english"))][:10]
        print("Topic {}: {}".format(idx, nearest_words))


gin.parse_config_file(args.config)
lda2vec = train()
