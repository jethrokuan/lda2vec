import tensorflow as tf

from tensorflow.contrib.training import HParams

baseline = HParams(
    learning_rate = 0.01,
    embedding_size = 256,
    num_topics = 20,
    temperature = 1.0,
    alpha = 0.7,
    vocabulary_size = 26863,
    negative_samples = 64
)
