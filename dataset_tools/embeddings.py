import numpy as np


def load_embeddings(file_path):
    """Loads embeddings into dictionary from a given file.

    Args:
       file_path: (str) path to pretrained embeddings

    Returns:

    """
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            w = values[0]
            vector = np.asarray(values[1:], dtype=np.float32)
            embeddings[w] = vector
    return embeddings
