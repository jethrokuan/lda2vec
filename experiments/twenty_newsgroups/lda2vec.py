"""Training lda2vec on twenty newsgroups."""

from utils.dirs import create_dirs
from models.lda2vec import Lda2Vec
import models.hparams.lda2vec as hparams
from trainers.lda2vec import Lda2VecTrainer
from utils.logger import Logger

import tensorflow as tf
BASE_DIR = "/Users/jethrokuan/Documents/Code/hash-lda2vec/lda2vec/"

CONFIG = {
    "summary_dir": "{}/summaries/".format(BASE_DIR),
    "checkpoint_dir": "{}/checkpoints/".format(BASE_DIR),
    "max_to_keep": 3,
    "batch_size": 50,
    "file_path": "experiments/twenty_newsgroups/train_data.csv",
    "num_epochs": 10000,
    "num_iter_per_epoch": 20,
    "num_documents": 11314,
}

create_dirs([CONFIG["summary_dir"], CONFIG["checkpoint_dir"]])

sess = tf.Session()

model = Lda2Vec(CONFIG, hparams.baseline)
logger = Logger(sess, CONFIG)
trainer = Lda2VecTrainer(sess, model, CONFIG, logger)
model.load(sess)
trainer.train()
