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
_CATEGORIES = ["soc.religion.christian", "sci.electronics", "comp.windows.x"]

parser = ArgumentParser()

parser.add_argument(
    "--output_path", help="Path to save the 20 newsgroups dataset.", required=True)

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

if os.path.exists(args.output_path):
    logger.info("File at {} exists. Exiting.".format(args.output_path))
    exit

logger.info("Downloading 20 newsgroups dataset...")

subsets = ["train", "test"]

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

for subset in subsets:
    texts = fetch_20newsgroups(subset=subset, categories=_CATEGORIES, remove=_NEWSGROUPS_REMOVE_SUBSET).data
    texts = [remove_from_line(line, _BAD_TEXT) for line in texts]
    file_path = os.path.join(args.output_path, "{}.txt".format(subset))

    with open(file_path, 'w+') as fp:
        for text in texts:
            fp.write("{}\n".format(text[:30]))

logger.info("20 newsgroups dataset written to {}".format(args.output_path))
logger.info("Program executed in {} seconds.".format(time.time() - start))
