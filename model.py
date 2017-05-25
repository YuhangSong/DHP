import numpy as np
import tensorflow as tf
import numpy as np
import random
import copy

import envs

import config

'''
Coder: YuhangSong
Description: main hyper-paramters for ct
'''
MaxNumLstmPerConsi = 6
LstmSize = 256
MaxActionSpace = 30
MinActionSpace = 2
FinalDropOutKeepProb = 1.0
IfUpdateFlowControl = False
ScaleToCut = 0.01

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]

        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])

        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters

        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.zeros_initializer,
                            collections=collections)

        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)

class LSTMPolicy(object):

    def __init__(self, consi_depth, ob_space, ac_space, env_id_str):

        '''convert env_id string to env_id num'''
        env_id_num = config.get_env_seq(config.game_dic_all).index(env_id_str)

        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))

        x = tf.expand_dims(flatten(x), 1)

        size = 256
        lstm = tf.contrib.rnn.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]

        c_in = tf.placeholder(tf.float32, [None, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [None, lstm.state_size.h])

        # c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        # h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])

        self.state_in = [c_in, h_in]

        self.step_size = tf.placeholder(tf.int32, [None])
        # self.step_size = tf.shape(self.x)[:1]

        state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=self.step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state

        x = tf.reshape(lstm_outputs, [-1, size])

        '''every game has its own pi and v'''
        logits_all = range(len(config.game_dic_all))
        vf_all = range(len(config.game_dic_all))
        for env_id_i_num in range(len(config.game_dic_all)):
            logits_all[env_id_i_num] = linear(x, config.get_env_ac_space(config.get_env_seq(config.game_dic_all)[env_id_i_num]), "action_"+str(env_id_i_num), normalized_columns_initializer(0.01))
            vf_all[env_id_i_num] = tf.reshape(linear(x, 1, "value_"+str(env_id_i_num), normalized_columns_initializer(1.0)), [-1])

        self.logits = logits_all[env_id_num]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.vf = vf_all[env_id_num]

        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h, self.step_size: [1]})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h, self.step_size: [1]})[0]
