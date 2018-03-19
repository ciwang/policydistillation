import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from core.deep_q_learning import DQN
from schedule import LinearExploration, LinearSchedule

class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model
        """
        # this information might be useful
        # here, typically, a state shape is (80, 80, 1)
        state_shape = list(self.env.observation_space.shape)
        num_actions = self.env.action_space.n

        self.s = tf.placeholder(tf.uint8, 
            shape=(None, state_shape[0], state_shape[1], state_shape[2] * self.config.state_history))
        self.a = tf.placeholder(tf.int32,
            shape=(None))
        self.r = tf.placeholder(tf.float32,
            shape=(None))
        self.sp = tf.placeholder(tf.uint8,
            shape=(None, state_shape[0], state_shape[1], state_shape[2] * self.config.state_history))
        self.done_mask = tf.placeholder(tf.bool,
            shape=(None))
        self.lr = tf.placeholder(tf.float32)

        if self.student:
            self.teacher_q = tf.placeholder(tf.float32,
                shape=(None, num_actions))


    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n
        out = state
        
        with tf.variable_scope(scope, reuse=reuse):
            out = layers.fully_connected(layers.flatten(out), num_actions, activation_fn=None)

        return out


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different set of weights. In tensorflow, we distinguish them
        with two different scopes. One for the target network, one for the
        regular network. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we need to update all the weights of the Q network 
        and assign them with the values from the regular network. Thus,
        what we need to do is to build a tf op, that, when called, will 
        assign all variables in the target network scope with the values of 
        the corresponding variables of the regular network scope.
    
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        
        q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
        target_q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_q_scope)
        assign_ops = [tf.assign(tqv, qv) for qv, tqv in zip(q_vars, target_q_vars)]
        self.update_target_op = tf.group(*assign_ops)


    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        num_actions = self.env.action_space.n

        not_done_mask = tf.abs(tf.cast(self.done_mask, tf.float32) - 1)
        q_samp = self.r + self.config.gamma * not_done_mask * tf.reduce_max(target_q, axis=1)
        a_indices = tf.one_hot(self.a, num_actions)
        q_sa = tf.reduce_sum(q * a_indices, axis=1)
        self.loss = tf.reduce_mean(tf.square(q_samp - q_sa))


    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        """

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        grads_and_vars = optimizer.compute_gradients(self.loss, var_list=var_list)
        if self.config.grad_clip:
            grads_and_vars = [(tf.clip_by_norm(gv[0], self.config.clip_val), gv[1]) for gv in grads_and_vars if gv[0] != None]
        self.train_op = optimizer.apply_gradients(grads_and_vars)
        self.grad_norm = tf.global_norm([gv[0] for gv in grads_and_vars])
    
