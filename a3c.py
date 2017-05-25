from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import LSTMPolicy
import six.moves.queue as queue
import scipy.signal
import threading
from socket import *
import struct
import time
from numpy import array
import subprocess
import copy
import config

DebugInModel = False
GAMMA = 0.99

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0):
    """
    given a rollout, compute its returns and the advantage
    """
    batch_si = np.asarray(rollout.states, dtype=np.float32)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    features_0 = np.asarray(rollout.features_0)
    features_1 = np.asarray(rollout.features_1)

    batch_adv_out = copy.deepcopy(batch_r)
    for i in range(len(batch_r)):
        batch_adv_out[i] = batch_adv[i]

    return Batch(batch_si, batch_a, batch_adv_out, batch_r, rollout.terminal, features_0, features_1)

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features_0", "features_1"])

class PartialRollout(object):
    """
    a piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features_0 = []
        self.features_1 = []

    def add(self, state, action, reward, value, terminal, features_0, features_1):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features_0 += [features_0]
        self.features_1 += [features_1]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features_0.extend(other.features_0)
        self.features_1.extend(other.features_1)

class RunnerThread(threading.Thread):
    """
    One of the key distinctions between a normal environment and a universe environment
    is that a universe environment is _real time_.  This means that there should be a thread
    that would constantly interact with the environment and tell it what to do.  This thread is here.
    """
    def __init__(self, env, env_id, policy, num_local_steps, log_thread):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.env_id = env_id
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.log_thread = log_thread

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.env_id, self.policy, self.num_local_steps, self.summary_writer, self.log_thread)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)

def env_runner(env, env_id, policy, num_local_steps, summary_writer, log_thread):
    """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    last_state = env.reset()
    last_features = policy.get_initial_features()
    length = 0
    rewards = 0

    while True:

        terminal_end = False
        rollout = PartialRollout()

        for _ in range(num_local_steps):
            fetched = policy.act(last_state, *last_features)
            action, value_, features = fetched[0], fetched[1], fetched[2:]

            # argmax to convert from one-hot
            state, reward, terminal, info = env.step(action.argmax())

            # collect the experience
            last_features_0 = last_features[0][0]
            last_features_1 = last_features[1][0]
            rollout.add(last_state, action, reward, value_, terminal, last_features_0, last_features_1)
            length += 1
            rewards += reward

            last_state = state
            last_features = features

            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    '''YuhangSong: here we add game id to compare different games in different graph'''
                    k = env_id + "/" + k
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if terminal or length >= timestep_limit:
                terminal_end = True
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                last_features = policy.get_initial_features()
                print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
                length = 0
                rewards = 0
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, *last_features)

        '''once we have enough experience, yield it, and have the TheradRunner place it on a queue'''
        yield rollout

class A3C(object):
    def __init__(self, consi_depth, env, env_id, task):
        """
        An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
        Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
        But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
        should be computed.
        """

        self.env = env
        self.task = task
        self.consi_depth = consi_depth
        self.env_id = env_id
        self.log_thread = False
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = LSTMPolicy(self.consi_depth, env.observation_space.shape, env.action_space.n, self.env_id)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.zeros_initializer(),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = LSTMPolicy(self.consi_depth, env.observation_space.shape, env.action_space.n, self.env_id)
                pi.global_step = self.global_step

            # self.env_id = 'PongDeterministic-v3'
            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")
            self.step_forward = tf.placeholder(tf.int32, [None], name="step_forward")

            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)

            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # ac will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            bs = tf.to_float(tf.shape(pi.x)[0])
            self.loss = pi_loss + 0.5 * vf_loss - entropy * 0.01

            # config.update_step represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            self.runner = RunnerThread(env, env_id, pi, config.update_step, self.log_thread)


            grads = tf.gradients(self.loss, pi.var_list)

            tf.summary.scalar(self.env_id+"/model/policy_loss", pi_loss / bs)
            tf.summary.scalar(self.env_id+"/model/value_loss", vf_loss / bs)
            tf.summary.scalar(self.env_id+"/model/entropy", entropy / bs)
            tf.summary.scalar(self.env_id+"/model/grad_global_norm", tf.global_norm(grads))
            tf.summary.scalar(self.env_id+"/model/var_global_norm", tf.global_norm(pi.var_list))

            self.summary_op = tf.summary.merge_all()
            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(self.step_forward)[0])

            # each worker has a different set of adam optimizer parameters
            opt = tf.train.AdamOptimizer(1e-4)
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            self.summary_writer = None
            self.local_steps = 0

    def start(self, sess, summary_writer):
        if(self.task!=0):
            print('this is not task 0, async from global network before start interaction and training')
            print('wait for the cheif thread before async')
            time.sleep(5)
            sess.run(self.sync)  # copy weights from shared to local
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

        print('before start mix exp')
        subprocess.call(["rm", "-r", config.mix_exp_temp_dir])
        subprocess.call(["mkdir", config.mix_exp_temp_dir])

    def pull_batch_from_queue(self):
        """
        self explanatory:  take a rollout from the queue of the thread runner.
        """
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
        process grabs a rollout that's been produced by the thread runner,
        and updates the parameters.  The update is then sent to the parameter
        server.
        """

        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=GAMMA, lambda_=1.0)

        should_compute_summary = self.task%config.num_workers_global == 0 and self.local_steps % 11 == 0

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        '''get batch_size'''
        batch_size = np.shape(batch.si)[0]

        '''load current one'''
        batch_si=batch.si
        batch_a=batch.a
        batch_adv=batch.adv
        batch_r=batch.r
        batch_features_0=batch.features_0
        batch_features_1=batch.features_1

        print('\tload ok for task:\t'+str(self.task)+'\tsized:\t'+str(np.shape(batch.si)[0]))
        step_forward = np.shape(batch.si)[0]

        if config.if_mix_exp is True:

            print("===========================mix exp==============================")

            '''save'''
            file_name = config.mix_exp_temp_dir + str(self.task) + '.npz'

            try:
                np.savez(file_name,
                         batch_si=batch.si,
                         batch_a=batch.a,
                         batch_adv=batch.adv,
                         batch_r=batch.r,
                         batch_features_0=batch.features_0,
                         batch_features_1=batch.features_1)

            except Exception:
                print('\tsave failed, go over\t')

            '''load exp from other exp'''
            for task_i in range(config.num_workers_total_global):

                if(task_i==self.task):
                    continue

                else:

                    file_name = config.mix_exp_temp_dir + str(task_i) + '.npz'

                    try:
                        data = np.load(file_name)

                    except Exception:
                        print('\tload failed for task:\t'+str(task_i)+'\tgo over\t')
                        continue

                    try:
                        '''temp data, for this could be wrong'''
                        batch_si_temp = np.concatenate((batch_si, data['batch_si']), axis=0)
                        batch_a_temp = np.concatenate((batch_a, data['batch_a']), axis=0)
                        batch_adv_temp = np.concatenate((batch_adv, data['batch_adv']), axis=0)
                        batch_r_temp = np.concatenate((batch_r, data['batch_r']), axis=0)
                        batch_features_0_temp = np.concatenate((batch_features_0, data['batch_features_0']), axis=0)
                        batch_features_1_temp = np.concatenate((batch_features_1, data['batch_features_1']), axis=0)

                        '''if not wrong'''
                        batch_si = batch_si_temp
                        batch_a = batch_a_temp
                        batch_adv = batch_adv_temp
                        batch_r = batch_r_temp
                        batch_features_0 = batch_features_0_temp
                        batch_features_1 = batch_features_1_temp

                        print('\tload ok for task:\t'+str(task_i)+'\tsized:\t'+str(np.shape(data['batch_si'])[0]))
                        data.close()

                    except Exception:
                        print('\twrong data in task:\t'+str(task_i)+'\tgo over\t')
                        data.close()
                        continue

            print('\tload all sized:\t'+str(np.shape(batch_si)[0]))
            print("==================================================================")
        '''
        ##################################################################
        '''

        feed_dict = {
            self.local_network.x: batch_si,
            self.ac: batch_a,
            self.adv: batch_adv,
            self.r: batch_r,
            self.local_network.state_in[0]: batch_features_0,
            self.local_network.state_in[1]: batch_features_1,
            self.local_network.step_size: [1]*( np.shape(batch_si)[0] ),
            self.step_forward: [1]*step_forward,
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1