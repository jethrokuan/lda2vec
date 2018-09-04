import os
import unittest

from dataset_tools.preprocess import preprocess
from dataset_tools.utils import read_file


class DatasetToolsUtilsTest(unittest.TestCase):
    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.test_text_file = os.path.join(os.path.join(dir_path, "test_data"),
                                           "lorem_ipsum.txt")

    def test_read_file(self):
        docs = read_file(self.test_text_file)
        tokenized_docs, id2word, word_counts = preprocess(docs,
                                                          min_word_count=10,
                                                          max_word_count=1000,
                                                          min_word_length=2,
                                                          max_doc_length=1000,
                                                          min_doc_length=10)
        self.assertEqual(len(tokenized_docs), 2)
        self.assertEqual(len(id2word.items()), 29)
        self.assertEqual(len(word_counts), 29)



if __name__ == "__main__":
    unittest.main()
