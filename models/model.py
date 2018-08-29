"""Base model class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class BaseModel(object):
    """Base model class."""
    def __init__(self, config):
        self.config = config
        self.cur_epoch_tensor = None
        self.increment_cur_epoch_op = None
        self.global_step_tensor = None
        self.increment_global_step_op = None

        self._init_global_step()
        self._init_current_epoch()

        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def _init_current_epoch(self):
        with tf.variable_scope("cur_epoch"):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name="cur_epoch")
            self.increment_cur_epoch_op = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def _init_global_step(self):
        with tf.variable_scope("global_step"):
            self.global_step_tensor = tf.Variable(0, trainable=False, name="global_step")
            self.increment_global_step_op = tf.assign(self.global_step_tensor, self.global_step_tensor + 1)

    def save(self, session):
        """Saves the checkpoint into the path defined in `config`.

        Args:
            session: the tf.Session() instance where the model was built.
        """
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)

    def load(self, sess):
        """Loads the latest checkpoint in the checkpoint directory.

        Args:
            session: the tf.Session() instance to restore the saved variables to."""
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            self.saver.restore(sess, latest_checkpoint)

    def build_graph(self):
        raise NotImplementedError("Sub-classes should implement this method.")
