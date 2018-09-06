from trainers.trainer import BaseTrainer

from tqdm import tqdm

import numpy as np

import tensorflow as tf
from utils.metrics import AverageMeter


class Word2VecTrainer(BaseTrainer):
    def __init__(self, sess, model, config, logger):
        super(Word2VecTrainer, self).__init__(sess, model, config, logger)
        self.dataset = tf.contrib.data.CsvDataset(
            self.config["file_path"],
            [tf.int32, tf.int32, tf.int32],
            header=True)
        self.dataset = self.dataset.shuffle(buffer_size=1000).repeat().batch(
            self.config["batch_size"])
        it = self.dataset.make_one_shot_iterator()
        self.next_batch = it.get_next()

    def train_epoch(self, epoch):
        loss_per_epoch = AverageMeter()
        losses = []
        for _ in tqdm(range(self.config["num_iter_per_epoch"])):
            loss = self.train_step()
            loss_per_epoch.update(loss)
        loss = np.mean(losses)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {"train/loss_per_epoch": loss_per_epoch.val}
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch = self.sess.run([self.next_batch])[0]
        context, _, target = batch

        feed_dict = {
            self.model.train_inputs: target,
            self.model.train_labels: context
        }
        _, loss = self.sess.run(
            [self.model.train_step, self.model.loss], feed_dict=feed_dict)

        return loss
