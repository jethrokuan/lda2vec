import os
import unittest

import numpy as np


from dataset_tools.preprocess import NlpPipeline
from dataset_tools.utils import read_file


class DatasetToolsUtilsTest(unittest.TestCase):
    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.test_text_file = os.path.join(os.path.join(dir_path, "test_data"),
                                           "lorem_ipsum.txt")
        self.docs = read_file(self.test_text_file)
        self.nlp_pipeline = NlpPipeline(self.docs, max_length=20)

    def test_pipeline(self):
        self.nlp_pipeline.tokenize()
        self.assertEqual(self.nlp_pipeline.tokenized_docs.shape, (4, 20))



if __name__ == "__main__":
    unittest.main()
