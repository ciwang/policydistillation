import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

from utils.general import get_logger
from utils.test_env import EnvTest
from schedule import LinearExploration, LinearSchedule
from linear import Linear
from natureqn import NatureQN

from config import testconfig_student as config
import logging

logging.basicConfig(level=logging.INFO)

CKPT_DIR = 'results/atari/teacher/model.weights/'

def initialize_teacher(session, model, train_dir, seed=42):
    tf.set_random_seed(seed)
    logging.info("Reading model parameters from %s" % train_dir)
    model.saver.restore(session, train_dir)
    logging.info('[Teacher] Num params: %d' % model.size)

class DistilledQN(NatureQN):
    def __init__(self, env, config, logger=None, student=True):
        teachermodel = NatureQN(env, config)
        teachermodel.initialize_basic()
        initialize_teacher(teachermodel.sess, teachermodel, CKPT_DIR)
        self.teachermodel = teachermodel
        super(DistilledQN, self).__init__(
            env, config, logger=logger, student=student)

    def add_loss_op(self, q, target_q):
        tau = 0.01
        eps = 0.00001

        ##############################

        # Loss functions for distillation

        # MSE
        self.loss = tf.losses.mean_squared_error(q, self.teacher_q)

        # NLL
        # self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=tf.argmax(self.teacher_q, axis=1), 
        #     logits=q))

        # KL
        # _p = tf.nn.softmax(self.teacher_q/tau, dim=1)+eps
        # _q = tf.nn.softmax(q, dim=1)+eps
        # self.loss = tf.reduce_sum(_p * tf.log(_p/_q))

"""
Use distilled Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    studentmodel = DistilledQN(env, config)
    studentmodel.run(exp_schedule, lr_schedule)
