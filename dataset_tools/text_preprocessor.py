"""Text Preprocessor class."""

import numpy as np


DEFAULT_CONFIG = {
    "max_length": 10000,
    "skip": "<SKIP>",
}

class TextPreprocessor(object):
    def __init__(self, documents, config):
        """Constructor for TextPreprocessor class.

        This class preprocesses text.

        Args:
            documents ([str]): list of strings, each string representing a document
            config (dict): dictionary for configuration
        """
        self.texts = texts
        self.num_docs = len(self.texts)
        self.tokenize()
        self.config = config

        # Initialize data
        self.data = np.zeros((self.num_docs, self.config.max_length), dtype=np.uint64)
