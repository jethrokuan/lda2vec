from tensorflow.contrib.training import HParams

baseline = HParams(
    learning_rate = 0.01,
    embedding_size = 256,
    vocabulary_size = 157164,
    negative_samples = 64
)
