"""Run a particular model.

Usage:
python run_trainer.py \
  --name lda2vec_twenty_newsgroups \
  --model lda2vec \
  --hparams "models.hparams.lda2vec.baseline" \
  --config "models.configs.lda2vec.twenty_newsgroups"
"""

import importlib
import tensorflow as tf

from utils.logger import Logger
from utils.dirs import create_dirs

from argparse import ArgumentParser

_MODELS = {
    "lda2vec": {
        "trainer": "trainers.lda2vec.Lda2VecTrainer",
        "model": "models.lda2vec.Lda2Vec",
    },
    "word2vec": {
        "trainer": "trainers.word2vec.Word2VecTrainer",
        "model": "models.word2vec.Word2Vec",
    }
}

parser = ArgumentParser()

parser.add_argument("--name", help="Experiment name.", required=True)

parser.add_argument(
    "--load_checkpoint",
    help="Flag to state whether to load a previous checkpoint.",
    action="store_true")

parser.add_argument(
    "--model",
    help="model type.",
    default="lda2vec",
    choices=["lda2vec", "word2vec"])

parser.add_argument(
    "--hparams", help="attribute path to hparams", required=True)

parser.add_argument("--config", help="attribute path to config", required=True)

args, _ = parser.parse_known_args()


def get_attribute_from_path(path):
    """Get named attribute from path.

    Args:
        path: full path to an attribute

    Returns: function or class or object matching path
             or None if path is invalid"""
    paths = path.rsplit('.', 1)
    if len(paths) != 2:
        return None

    module = importlib.import_module(paths[0])
    return getattr(module, paths[1])


config = get_attribute_from_path(args.config)
hparams = get_attribute_from_path(args.hparams)

create_dirs([config["summary_dir"], config["checkpoint_dir"]])

sess = tf.Session()
selected_model = _MODELS[args.model]

Model = get_attribute_from_path(selected_model["model"])
Trainer = get_attribute_from_path(selected_model["trainer"])

logger = Logger(sess, config)
model = Model(config, hparams)
trainer = Trainer(sess, model, config, logger)

if args.load_checkpoint:
    model.load(sess)
elif len(os.listdir(config["summary_dir"])) != 0 or len(
        os.listdir(config["summary_dir"])) != 0:
    raise Exception(
        "You might overwrite your old model configuration! exiting...")

trainer.train()
