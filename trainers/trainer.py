"Base trainer class."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class BaseTrainer(object):
    def __init__(self, sess, model, config, logger, data_loader=None):
        """Constructor for the BaseTrainer class.

        Args:
           sess: a tf.Session() instance.
           model: a model, that inherits BaseModel.
           config: configuration.
           logger: the logger used to write values to Tensorboard.
           data_loader: The data loader. Defaults to None.
        """

        self.sess = sess
        self.model = model
        self.config = config
        self.logger = logger
        self.data_loader = data_loader

        self.sess.run(tf.global_variables_initializer())

    def train(self):
        """This is the main loop for training the model."""
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess),
                               self.config["num_epochs"] + 1):
            self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_op)

    def train_epoch(self, epoch):
        """Runs one epoch of training.

        This method is responsible for:
          1. Calling the train step
          2. Adding any summaries to the logger

        Args:
          epoch: the current epoch.

        Returns:
          None
        """
        raise NotImplementedError("Sub-class should implement this method.")

    def train_step(self):
        """Runs the tensorflow session.

        Returns:
            summary metrics
        """
        raise NotImplementedError("Sub-class should implement this method.")
