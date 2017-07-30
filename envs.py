import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from gym.spaces.box import Box
import numpy as np
import numpy
import gym
from gym import spaces
import logging
import universe
from universe import vectorized
from universe.wrappers import BlockingReset, GymCoreAction, EpisodeID, Unvectorize, Vectorize, Vision, Logger
from universe import spaces as vnc_spaces
from universe.spaces.vnc_event import keycode
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt, log
import math
import copy
from mpl_toolkits.mplot3d import Axes3D
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq,kmeans,whiten

import subprocess
import urllib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from vrplayer import get_view
from move_view_lib import move_view

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()

class env_f():
    def __init__(self, env_id, task, subject=None, summary_writer=None):

        self._episode_reward = 0
        self._episode_length = 0

        class nnn():
            def __init__(self, n):
                self.n = n
        import config
        self.action_space = nnn(config.direction_num)

        self.env_id = env_id
        import envs_li
        self.env_li = envs_li.env_li(env_id=env_id,
                                     task=task,
                                     subject=subject,
                                     summary_writer=summary_writer)

        '''observation_space'''
        from config import observation_space
        self.observation_space = observation_space

        '''warper to meet origin env'''

        '''env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')'''

        class spec():
            def __init__(self, env_li):
                class tags():
                    def __init__(self, env_li):
                        self.env_li = env_li
                    def get(self,get_str):
                        if get_str=='wrapper_config.TimeLimit.max_episode_steps':
                            return self.env_li.step_total
                        else:
                            print(s)
                self.tags = tags(env_li)
        self.spec = spec(self.env_li)

    def reset(self):
        return self.env_li.reset()

    def step(self, action, v):

        from config import mode
        if mode is 'off_line':
            observation, reward, done, cur_cc, max_cc, v_lable = self.env_li.step(action, v)
        elif mode is 'on_line':
            observation, reward, done, cur_cc, max_cc, v_lable, predicting = self.env_li.step(action, v)

        to_log = {}

        if done:

            logger.info('Episode terminating: episode_reward=%s episode_length=%s', self._episode_reward, self._episode_length)

            to_log["global/episode_reward"] = self._episode_reward
            to_log["global/cur_cc"] = cur_cc
            to_log["global/max_cc"] = max_cc
            # to_log["global/episode_length"] = self._episode_length

            self._episode_reward = 0
            self._episode_length = 0

        else:

            self._episode_reward += reward
            self._episode_length += 1

        from config import mode
        if mode is 'off_line':
            return observation, reward, done, to_log, v_lable
        elif mode is 'on_line':
            return observation, reward, done, to_log, v_lable, predicting


def create_env(env_id, client_id, remotes, task=0, subject=None, summary_writer=None, **kwargs):
    import config
    if config.project is 'g':
        spec = gym.spec(env_id)
        if spec.tags.get('flashgames', False):
            return create_flash_env(env_id, client_id, remotes, **kwargs)
        elif spec.tags.get('atari', False) and spec.tags.get('vnc', False):
            return create_vncatari_env(env_id, client_id, remotes, **kwargs)
        else:
            # Assume atari.
            assert "." not in env_id  # universe environments have dots in names.
            return create_atari_env(env_id)
    elif config.project is 'f':
        return env_f(env_id = env_id,
                     task = task,
                     subject = subject,
                     summary_writer=summary_writer)

def create_flash_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Vision(env)
    env = Logger(env)
    env = BlockingReset(env)

    reg = universe.runtime_spec('flashgames').server_registry
    height = reg[env_id]["height"]
    width = reg[env_id]["width"]
    env = CropScreen(env, height, width, 84, 18)
    env = FlashRescale(env)

    keys = ['left', 'right', 'up', 'down', 'x']
    env = DiscreteToFixedKeysVNCActions(env, keys)
    env = EpisodeID(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    env.configure(fps=5.0, remotes=remotes, start_timeout=15 * 60, client_id=client_id,
                  vnc_driver='go', vnc_kwargs={
                    'encoding': 'tight', 'compress_level': 0,
                    'fine_quality_level': 50, 'subsample_level': 3})
    return env

def create_vncatari_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Vision(env)
    env = Logger(env)
    env = BlockingReset(env)
    env = GymCoreAction(env)
    env = AtariRescale42x42(env)
    env = EpisodeID(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)

    logger.info('Connecting to remotes: %s', remotes)
    fps = env.metadata['video.frames_per_second']
    env.configure(remotes=remotes, start_timeout=15 * 60, fps=fps, client_id=client_id)
    return env

def create_atari_env(env_id):
    env = gym.make(env_id)
    env = Vectorize(env)
    env = AtariRescale42x42(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    return env

def DiagnosticsInfo(env, *args, **kwargs):
    return vectorized.VectorizeFilter(env, DiagnosticsInfoI, *args, **kwargs)

class DiagnosticsInfoI(vectorized.Filter):
    def __init__(self, log_interval=503):
        super(DiagnosticsInfoI, self).__init__()

        self._episode_time = time.time()
        self._last_time = time.time()
        self._local_t = 0
        self._log_interval = log_interval
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        self._num_vnc_updates = 0
        self._last_episode_id = -1

    def _after_reset(self, observation):
        logger.info('Resetting environment')
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        return observation

    def _after_step(self, observation, reward, done, info):
        to_log = {}
        if self._episode_length == 0:
            self._episode_time = time.time()

        self._local_t += 1
        if info.get("stats.vnc.updates.n") is not None:
            self._num_vnc_updates += info.get("stats.vnc.updates.n")

        if self._local_t % self._log_interval == 0:
            cur_time = time.time()
            elapsed = cur_time - self._last_time
            fps = self._log_interval / elapsed
            self._last_time = cur_time
            cur_episode_id = info.get('vectorized.episode_id', 0)
            a=1# to_log["diagnostics/fps"] = fps
            if self._last_episode_id == cur_episode_id:
                a=1# to_log["diagnostics/fps_within_episode"] = fps
            self._last_episode_id = cur_episode_id
            if info.get("stats.gauges.diagnostics.lag.action") is not None:
                a=1# to_log["diagnostics/action_lag_lb"] = info["stats.gauges.diagnostics.lag.action"][0]
                a=1# to_log["diagnostics/action_lag_ub"] = info["stats.gauges.diagnostics.lag.action"][1]
            if info.get("reward.count") is not None:
                a=1# to_log["diagnostics/reward_count"] = info["reward.count"]
            if info.get("stats.gauges.diagnostics.clock_skew") is not None:
                a=1# to_log["diagnostics/clock_skew_lb"] = info["stats.gauges.diagnostics.clock_skew"][0]
                a=1# to_log["diagnostics/clock_skew_ub"] = info["stats.gauges.diagnostics.clock_skew"][1]
            if info.get("stats.gauges.diagnostics.lag.observation") is not None:
                a=1# to_log["diagnostics/observation_lag_lb"] = info["stats.gauges.diagnostics.lag.observation"][0]
                a=1# to_log["diagnostics/observation_lag_ub"] = info["stats.gauges.diagnostics.lag.observation"][1]

            if info.get("stats.vnc.updates.n") is not None:
                a=1# to_log["diagnostics/vnc_updates_n"] = info["stats.vnc.updates.n"]
                a=1# to_log["diagnostics/vnc_updates_n_ps"] = self._num_vnc_updates / elapsed
                self._num_vnc_updates = 0
            if info.get("stats.vnc.updates.bytes") is not None:
                a=1# to_log["diagnostics/vnc_updates_bytes"] = info["stats.vnc.updates.bytes"]
            if info.get("stats.vnc.updates.pixels") is not None:
                a=1# to_log["diagnostics/vnc_updates_pixels"] = info["stats.vnc.updates.pixels"]
            if info.get("stats.vnc.updates.rectangles") is not None:
                a=1# to_log["diagnostics/vnc_updates_rectangles"] = info["stats.vnc.updates.rectangles"]
            if info.get("env_status.state_id") is not None:
                a=1# to_log["diagnostics/env_state_id"] = info["env_status.state_id"]

        if reward is not None:
            self._episode_reward += reward
            if observation is not None:
                self._episode_length += 1
            self._all_rewards.append(reward)

        if done:
            logger.info('Episode terminating: episode_reward=%s episode_length=%s', self._episode_reward, self._episode_length)
            total_time = time.time() - self._episode_time
            '''
            YuhangSong: only this two we are intrested in
            '''
            to_log["global/episode_reward"] = self._episode_reward
            to_log["global/episode_length"] = self._episode_length
            a=1# to_log["global/episode_time"] = total_time
            a=1# to_log["global/reward_per_time"] = self._episode_reward / total_time
            self._episode_reward = 0
            self._episode_length = 0
            self._all_rewards = []

        # print(observation)
        # print(np.shape(observation))
        # print(reward)
        # print(done)
        # print(to_log)
        '''
        (42, 42, 1)
        0.0
        False
        {}
        '''

        return observation, reward, done, to_log

def _process_frame42(frame):
    frame = frame[34:34+160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame

class AtariRescale42x42(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0, 255, [42, 42, 1])

    def _observation(self, observation_n):
        return [_process_frame42(observation) for observation in observation_n]

class FixedKeyState(object):
    def __init__(self, keys):
        self._keys = [keycode(key) for key in keys]
        self._down_keysyms = set()

    def apply_vnc_actions(self, vnc_actions):
        for event in vnc_actions:
            if isinstance(event, vnc_spaces.KeyEvent):
                if event.down:
                    self._down_keysyms.add(event.key)
                else:
                    self._down_keysyms.discard(event.key)

    def to_index(self):
        action_n = 0
        for key in self._down_keysyms:
            if key in self._keys:
                # If multiple keys are pressed, just use the first one
                action_n = self._keys.index(key) + 1
                break
        return action_n

class DiscreteToFixedKeysVNCActions(vectorized.ActionWrapper):
    def __init__(self, env, keys):
        super(DiscreteToFixedKeysVNCActions, self).__init__(env)

        self._keys = keys
        self._generate_actions()
        self.action_space = spaces.Discrete(len(self._actions))
        self.key_state = FixedKeyState(keys)

    def _generate_actions(self):
        self._actions = []
        for key in [''] + self._keys:
            cur_action = []
            for cur_key in self._keys:
                cur_action.append(vnc_spaces.KeyEvent.by_name(cur_key, down=cur_key == key))
            self._actions.append(cur_action)

    def _action(self, action_n):
        # Each action might be a length-1 np.array. Cast to int to
        # avoid warnings.
        return [self._actions[int(action)] for action in action_n]

class CropScreen(vectorized.ObservationWrapper):
    """Crops out a [height]x[width] area starting from (top,left) """
    def __init__(self, env, height, width, top=0, left=0):
        super(CropScreen, self).__init__(env)
        self.height = height
        self.width = width
        self.top = top
        self.left = left
        self.observation_space = Box(0, 255, shape=(height, width, 3))

    def _observation(self, observation_n):
        return [ob[self.top:self.top+self.height, self.left:self.left+self.width, :] if ob is not None else None
                for ob in observation_n]

def _process_frame_flash(frame):
    frame = cv2.resize(frame, (200, 128))
    frame = frame.mean(2).astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [128, 200, 1])
    return frame

class FlashRescale(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(FlashRescale, self).__init__(env)
        self.observation_space = Box(0, 255, [128, 200, 1])

    def _observation(self, observation_n):
        return [_process_frame_flash(observation) for observation in observation_n]
