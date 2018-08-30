"""Class for fetching 20 newsgroups dataset.

This script only downloads a subset of the 20 newsgroups dataset,
and removes a small set of bad words. The file is written to be further
processed by a downstream pipeline.
"""

import os
from argparse import ArgumentParser
from sklearn.datasets import fetch_20newsgroups

import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_BAD_TEXT = set(["ax>", '`@("', '---', '===', '^^^', "AX>", "GIZ"])
_NEWSGROUPS_REMOVE_SUBSET = ('headers', 'footers', 'quotes')

parser = ArgumentParser()

parser.add_argument(
    "--file_path", help="Path to save the 20 newsgroups dataset.", required=True)

args, _ = parser.parse_known_args()

start = time.time()

def remove_from_line(line, remove_set):
    """Removes words in remove_set from line.

    Args:
       line: Input line.
       remove_set: set of words to remove from line.

    Returns:
       Line with words removed, if any."""
    return " ".join(w for w in line.split() if not any(t in w for t in remove_set))

if os.path.exists(args.file_path):
    logger.info("File at {} exists. Exiting.".format(args.file_path))
    exit

logger.info("Downloading 20 newsgroups dataset...")
os.makedirs(os.path.dirname(args.file_path), exist_ok=True)
texts = fetch_20newsgroups(subset='train', remove=_NEWSGROUPS_REMOVE_SUBSET).data
texts = [remove_from_line(line, _BAD_TEXT) for line in texts]

with open(args.file_path, 'w+') as fp:
    for text in texts:
        fp.write("{}\n".format(text))

logger.info("20 newsgroups dataset written to {}".format(args.file_path))
logger.info("Program executed in {} seconds.".format(time.time() - start))
