import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from schedule import LinearExploration, LinearSchedule
from linear import Linear

from config import testconfig_teacher as config

class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
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
        # if self.student == True:
        #     return super(NatureQN, self).get_q_values_op(
        #         state, scope, reuse)

        # this information might be useful
        num_actions = self.env.action_space.n
        out = state
        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

              you may find the section "model architecture" of the appendix of the 
              nature paper particulary useful.

              store your result in out of shape = (batch_size, num_actions)

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param
              make sure to flatten() the tensor before connecting it to fully connected layers 

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)

        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################ 

        # compress the student network
        size1, size2, size3, size4 = (16, 16, 16, 128) if self.student else (32, 64, 64, 512)

        with tf.variable_scope(scope, reuse=reuse):
            conv1 = layers.conv3d(inputs=out, num_outputs=size1, kernel_size=[8,8], stride=4) #20
            conv2 = layers.conv3d(inputs=conv1, num_outputs=size2, kernel_size=[4,4], stride=2) #10
            conv3 = layers.conv3d(inputs=conv2, num_outputs=size3, kernel_size=[3,3], stride=1) #10
            hidden = layers.fully_connected(layers.flatten(conv3), size4)
            out = layers.fully_connected(hidden, num_actions, activation_fn=None)

        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
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
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
