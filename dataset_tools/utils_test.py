import os
import unittest

from dataset_tools.utils import read_file


class DatasetToolsUtilsTest(unittest.TestCase):
    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.test_text_file = os.path.join(os.path.join(dir_path, "test_data"),
                                           "short.txt")

    def test_read_file(self):
        lines = read_file(self.test_text_file)
        self.assertEqual(lines, ["hello", "there"])


if __name__ == "__main__":
    unittest.main()
