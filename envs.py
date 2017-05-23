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
from vrplayer import get_view
from move_view_lib import move_view
import subprocess
import urllib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from cc import calc_score
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()

'''for test_scan_path'''
if_training = True
if_log_scan_path_global = False
if_log_cc_global = False

'''for measure consistence'''
fov_degree = 6
no_moving_gate = 0.0001
compute_lon_inter = fov_degree / 2
compute_lat_inter = fov_degree / 2
frame_gate = 20
MaxCenterNum = 4

NumDirectionForCluster = 8
DirectionInter = 360 / NumDirectionForCluster

'''for env'''
data_tensity = 10
far_away_gate_in_degree = 103/4.0

view_range_lon = 103
view_range_lat = 60
ActionSpace = 8
terminal_type_meet_end = 1
terminal_type_far_away = 2

final_discount = 4
final_discounted_config = 10**(-final_discount)
v_relative_discounted_config = 0.5

def fixation2salmap(fixation, mapwidth, mapheight):
    W = 2 #Full width at half max in visual degrees of angle for the Gaussian
    my_sigma = W/(2*math.sqrt(2*math.log(2)))
    fixation_total = np.shape(fixation)[0]
    x_degree_per_pixel = 360.0 / mapwidth
    y_degree_per_pixel = 180.0 / mapheight
    salmap = np.zeros((mapwidth, mapheight))
    for x in range(mapwidth):
        for y in range(mapheight):
            cur_lon = x * x_degree_per_pixel - 180.0
            cur_lat = y * y_degree_per_pixel - 90.0
            for fixation_count in range(fixation_total):
                cur_fixation_lon = fixation[fixation_count][0]
                cur_fixation_lat = fixation[fixation_count][1]
                distance_to_cur_fixation = haversine(lon1=cur_lon,
                                                     lat1=cur_lat,
                                                     lon2=cur_fixation_lon,
                                                     lat2=cur_fixation_lat)
                sal = math.exp(-1.0 / 2.0 * (distance_to_cur_fixation**2) / (my_sigma**2))
                salmap[x, y] += sal
    salmap = salmap * ( 1.0 / np.amax(salmap) )
    return salmap

class env_li():

    '''
    Function: env interface for ff
    Coder: syh
    Status: checking
    '''

    def __init__(self, id_, observation_space, select=0, if_log_scan_path=False, if_log_cc=False):

        self.if_log_scan_path = if_log_scan_path
        if(self.if_log_scan_path==True):
            print("if_log_scan_path is True")
        self.if_log_cc = if_log_cc
        if(self.if_log_cc==True):
            print("if_log_cc is True")
            self.cc_count = 0

        '''observation_space'''
        self.observation_space = observation_space

        '''get id contains only name of the video'''
        self.id = id_

        '''get subjects_total'''
        data = sio.loadmat('../../vr/FULLdata_per_video_frame.mat')[self.id]
        self.subjects_total = get_num_subjects(data=data)

        '''if this worker render'''
        self.worker_id = select
        if(self.worker_id==0):
            self.is_render=False
        else:
            self.is_render=False

        '''set all temp dir for this worker'''
        self.temp_dir = "temp_w_" + str(self.worker_id) + "/"
        '''clear temp dir for this worker'''
        subprocess.call(["rm", "-r", self.temp_dir])
        subprocess.call(["mkdir", self.temp_dir])

        '''load config'''
        self.config()

        if(self.if_log_cc==True):
            self.cc_count = 0
            # self.cc_count_to = self.subjects_total * 8
            self.cc_count_to = 1
            self.agent_result_saver = np.zeros((self.step_total, self.cc_count_to, 2))

            '''salmap'''
            self.salmap_width = 360
            self.salmap_height = 180

        '''reset'''
        self.observation = self.reset(terminal_type_meet_end)

    def get_observation(self):

        '''interface to get view'''
        self.cur_observation = get_view(input_width=self.video_size_width,
                                        input_height=self.video_size_heigth,
                                        view_fov_x=self.view_range_lon,
                                        view_fov_y=self.view_range_lat,
                                        cur_frame=self.cur_frame,
                                        is_render=self.is_render,
                                        output_width=np.shape(self.observation_space)[0],
                                        output_height=np.shape(self.observation_space)[1],
                                        view_center_lon=self.cur_lon,
                                        view_center_lat=self.cur_lat,
                                        # output_width=400,
                                        # output_height=400,
                                        # view_center_lon=self.cur_step%8*45.0-180.0,
                                        # view_center_lat=self.cur_step%4*45.0-90.0,
                                        temp_dir=self.temp_dir,
                                        file_='../../vr/' + self.id + '.yuv')

    def config(self):

        '''function to load config'''
        print("=================config=================")

        '''load in mat data of head movement'''
        matfn = '../../vr/FULLdata_per_video_frame.mat'
        data_all = sio.loadmat(matfn)
        data = data_all[self.id]

        print("env set to: "+str(self.id))

        '''frame bug'''
        '''some bug in the frame read for some video,='''
        if(self.id=='Dubai'):
            self.frame_bug_offset = 540
        elif(self.id=='MercedesBenz'):
            self.frame_bug_offset = 10
        elif(self.id=='Cryogenian'):
            self.frame_bug_offset = 10
        else:
            self.frame_bug_offset = 0

        '''get subjects'''
        self.subjects_total, self.data_total, self.subjects, _ = get_subjects(data,0)

        '''init video and get paramters'''
        video = cv2.VideoCapture('../../vr/' + self.id + '.mp4')
        self.frame_per_second = video.get(cv2.cv.CV_CAP_PROP_FPS)
        self.frame_total = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        self.video_size_width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        self.video_size_heigth = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        self.second_total = self.frame_total / self.frame_per_second
        self.data_per_frame = self.data_total / self.frame_total

        '''compute step lenth from data_tensity'''
        self.second_per_step = max(data_tensity/self.frame_per_second, data_tensity/self.data_per_frame/self.frame_per_second)
        self.frame_per_step = self.frame_per_second * self.second_per_step
        self.data_per_step = self.data_per_frame * self.frame_per_step

        self.step_total = int(self.data_total / self.data_per_step) + 1

        '''set fov range'''
        self.view_range_lon = view_range_lon
        self.view_range_lat = view_range_lat

    def reset(self, terminal_type, frame=None):

        '''reset cur_step and cur_data no matter what terminal_type is'''
        self.cur_step = 0
        self.cur_data = 0

        '''if terminal_type is terminal_type_meet_end'''
        if(terminal_type==terminal_type_meet_end):

            print("terminal_type_meet_end")

            '''reset cur_frame'''
            self.cur_frame = 0

            '''reset cur_lon and cur_lat to one of the subjects start point'''
            subject_dic_code = []
            for i in range(self.subjects_total):
                subject_dic_code += [i]
            subject_code = np.random.choice(a=subject_dic_code)
            self.cur_lon = self.subjects[subject_code].data_frame[0].p[0]
            self.cur_lat = self.subjects[subject_code].data_frame[0].p[1]

            '''set observation_now to the first frame'''
            self.get_observation()

            self.last_observation = None

        elif(terminal_type==terminal_type_far_away):

            print('!!!please specify terminal_type!!!')

        else:

            print('!!!please specify terminal_type!!!')

        return self.cur_observation

    def step(self, action):

        '''log_scan_path'''
        if(self.if_log_scan_path==True):
            plt.figure(str(self.id)+'_scan_path')
            subject_code=1
            if(self.cur_lon>180):
                draw_lon = self.cur_lon - 360.0
            else:
                draw_lon = self.cur_lon
            plt.scatter(draw_lon, self.cur_lat, c='r')
            # plt.scatter(self.subjects[subject_code].data_frame[self.cur_data].p[0], self.subjects[subject_code].data_frame[self.cur_data].p[1], c='b')
            plt.scatter(-180, -90)
            plt.scatter(-180, 90)
            plt.scatter(180, -90)
            plt.scatter(180, 90)
            plt.pause(0.1)

        '''log_cc'''
        if(self.if_log_cc==True):

            if(self.cur_lon>180):
                draw_lon = self.cur_lon - 360.0
            else:
                draw_lon = self.cur_lon
            draw_lat = self.cur_lat

            self.agent_result_saver[self.cur_step, self.cc_count, 0] = draw_lon
            self.agent_result_saver[self.cur_step, self.cc_count, 1] = draw_lat

        '''varible for record state is stored, for they will be updated'''
        self.last_step = self.cur_step
        self.last_data = self.cur_data
        self.last_observation = self.cur_observation
        self.last_lon = self.cur_lon
        self.last_lat = self.cur_lat
        self.last_frame = self.cur_frame

        '''update cur_step'''
        self.cur_step += 1

        '''update cur_data'''
        self.cur_data = int(round((self.cur_step)*self.data_per_step))
        if(self.cur_data>=self.data_total):
            update_data_success = False
        else:
            update_data_success = True

        '''update cur_frame'''
        self.cur_frame = int(round((self.cur_step)*self.frame_per_step))
        if(self.cur_frame>=(self.frame_total-self.frame_bug_offset)):
            update_frame_success = False
        else:
            update_frame_success = True

        '''if any of update frame or update data is failed'''
        if(update_frame_success==False)or(update_data_success==False):

            '''terminating'''
            terminal_type = terminal_type_meet_end
            self.reset(terminal_type=terminal_type)
            reward = 0.0
            done = True

            if(self.if_log_cc==True):

                self.cc_count += 1
                if(self.cc_count>=self.cc_count_to):

                    '''compute cc'''
                    cc_dic = []
                    for draw_step_count in range(self.step_total):
                        draw_cur_data = int(round((draw_step_count)*self.data_per_step))

                        '''plot fixation'''
                        plt.figure(str(self.id)+'_fixation')
                        plt.clf()
                        for draw_cc_count in range(self.cc_count_to):
                            plt.scatter(self.agent_result_saver[draw_step_count, draw_cc_count, 0], self.agent_result_saver[draw_step_count, draw_cc_count, 1], c='r')

                        for draw_subjects_count in range(self.subjects_total):
                            plt.scatter(self.subjects[draw_subjects_count].data_frame[draw_cur_data].p[0], self.subjects[draw_subjects_count].data_frame[draw_cur_data].p[1], c='b')
                        plt.scatter(-180, -90)
                        plt.scatter(-180, 90)
                        plt.scatter(180, -90)
                        plt.scatter(180, 90)
                        plt.pause(0.1)

                        '''generate ground-truth salmap'''
                        groundtruth_fixation = np.zeros((self.subjects_total, 2))
                        for get_fixation_subjects_count in range(self.subjects_total):
                            groundtruth_fixation[get_fixation_subjects_count, 0] = self.subjects[get_fixation_subjects_count].data_frame[draw_cur_data].p[0]
                            groundtruth_fixation[get_fixation_subjects_count, 1] = self.subjects[get_fixation_subjects_count].data_frame[draw_cur_data].p[1]
                        groundtruth_salmap = fixation2salmap(groundtruth_fixation, self.salmap_width, self.salmap_height)

                        '''generate predicted salmap'''
                        predicted_salmap = fixation2salmap(self.agent_result_saver[draw_step_count], self.salmap_width, self.salmap_height)

                        cc = calc_score(groundtruth_salmap, predicted_salmap)

                        print("cc for step "+str(draw_step_count)+" is "+str(cc))

                        f = open(str(self.id)+'_cc_on_frame.txt','a')

                        print_string = '\t'

                        print_string += 'draw_step_count' + '\t'
                        print_string += str(draw_step_count) + '\t'

                        print_string += 'cc' + '\t'
                        print_string += str(cc) + '\t'

                        print_string += '\n'

                        f.write(print_string)
                        f.close()

                        cc_dic += [cc]

                    cc_averaged = sum(cc_dic)/len(cc_dic)

                    print("cc for this test round is "+str(cc_averaged))

                    f = open('average_cc_on_videos.txt','a')

                    print_string = '\t'

                    print_string += 'id' + '\t'
                    print_string += str(self.id) + '\t'

                    print_string += 'cc_averaged' + '\t'
                    print_string += str(cc_averaged) + '\t'

                    print_string += '\n'

                    f.write(print_string)
                    f.close()

                    print("test cc over, the programe will terminate")
                    print(s)

                    self.cc_count = 0

        else:

            '''get reward and v from last state'''
            last_prob, distance_per_data = get_prob(lon=self.last_lon,
                                              lat=self.last_lat,
                                              theta=action * 45.0,
                                              subjects=self.subjects,
                                              subjects_total=self.subjects_total,
                                              cur_data=self.last_data)

            '''rescale'''
            distance_per_step = distance_per_data * self.data_per_step

            '''convert v to degree'''
            degree_per_step = distance_per_step / math.pi * 180.0

            '''move view, update cur_lon and cur_lat'''
            self.cur_lon, self.cur_lat = move_view(cur_lon=self.last_lon,
                                                   cur_lat=self.last_lat,
                                                   direction=action,
                                                   degree_per_step=degree_per_step)

            '''update observation_now'''
            self.get_observation()

            '''produce output'''
            terminal_type = None
            reward = last_prob
            done = False

        return self.cur_observation, reward, done, terminal_type

class env_f():
    def __init__(self, log_interval=503, id_ = 'Movie/Help', select=0, if_log_scan_path=False, if_log_cc=False):

        self._episode_reward = 0
        self._episode_length = 0
        self._episode_resettimes = 1

        self.observation_space = np.zeros((42, 42, 1))
        self.action_space = nnn(ActionSpace)
        self.id_f = id_
        self.env_li = env_li(id_,
                             self.observation_space,
                             select,
                             if_log_scan_path=if_log_scan_path,
                             if_log_cc=if_log_cc)

    def reset(self, terminal_type):
        return self.env_li.reset(terminal_type)

    def step(self, action):

        observation, reward, done, terminal_type = self.env_li.step(action)

        to_log = {}

        if done and terminal_type is terminal_type_meet_end:

            logger.info('Episode terminating: episode_reward=%s episode_length=%s', self._episode_reward, self._episode_length)

            to_log["global/episode_reward"] = self._episode_reward
            # to_log["global/episode_length"] = self._episode_length
            # to_log["global/episode_resettimes"] = self._episode_resettimes

            self._episode_reward = 0
            self._episode_length = 0
            self._episode_resettimes = 1

        else:

            if terminal_type is terminal_type_far_away:
                self._episode_resettimes += 1

            self._episode_reward += reward
            self._episode_length += 1

        return observation, reward, done, to_log, terminal_type, self.id_f

def create_env(env_id, client_id, remotes, id_ff = 'Movie/Help', select=0, if_log_scan_path=False, if_log_cc=False, **kwargs):

    if(env_id=='FfDeterministic-v3'):

        return env_f(id_ = id_ff,
                     select=select,
                     if_log_scan_path=if_log_scan_path,
                     if_log_cc=if_log_cc)

    else:

        spec = gym.spec(env_id)

        if spec.tags.get('flashgames', False):
            return create_flash_env(env_id, client_id, remotes, **kwargs)
        elif spec.tags.get('atari', False) and spec.tags.get('vnc', False):
            return create_vncatari_env(env_id, client_id, remotes, **kwargs)
        else:
            # Assume atari.
            assert "." not in env_id  # universe environments have dots in names.
            return create_atari_env(env_id)

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

class tags_f():
    def __init__(self, env):

        self.env = env

    def set_step_limit(self, step_limit):

        self.step_limit = step_limit

    def get(self, name):

        return self.step_limit

class spec_f():
    def __init__(self, env):
        self.tags = tags_f(env)

class nnn():
    def __init__(self, n):
        self.n = n

# def haversine(lon1, lat1, lon2, lat2):
#
#     '''
#     Function: Calculate the great circle distance between two points on the sphere
#     Coder: syh
#     Status: unchecked
#     '''
#
#     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
#     return math.acos( math.cos(lat1) * math.cos(lat2) * math.cos(lon1-lon2) + math.sin(lat1) * math.sin(lat2) )

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 1.0
    return c * r

def lonlat2Mercator(lon, lat):

    '''
    Function: convert lon-lat to Mercator cardination
    Coder: syh
    Status: unchecked
    '''

    x = lon * 20037508.34 / 180
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0

    return x, y

def T_trustworth_transfer(lon, lat, data_frame, thegma=0.7):

    '''
    Function: trustworthy transfer for lon-lat domain
    Coder: syh
    Status: unchecked
    '''

    '''compute distance on sphere'''
    distance = haversine(lon1=lon,
                         lat1=lat,
                         lon2=data_frame.p[0],
                         lat2=data_frame.p[1])

    '''guassion trustworthy transfer'''
    prob = data_frame.t * math.exp(-1.0 / 2.0 * (distance**2) / (thegma**2))

    return prob

def G_trustworth_transfer(lon, lat, theta, data_frame, thegma=10):

    '''
    Function: trustworthy transfer for theta domain
    Coder: syh
    Status: unchecked
    '''

    prob = T_trustworth_transfer(lon            = lon,
                                 lat            = lat,
                                 data_frame     = data_frame)

    '''compute distance on sphere'''
    distance = abs(theta - data_frame.theta)
    if(distance>180):
        distance = distance - 180

    '''guassion trustworthy transfer'''
    prob = prob * math.exp(-1.0 / 2.0 * (distance**2) / (thegma**2))

    return prob

def get_transfered_data(lon, lat, theta, data_frame, max_distance_on_position=1.0*math.pi, max_distance_on_degree=180.0, final_discounted=final_discounted_config):

    distance_on_position = haversine(lon1=lon,
                                     lat1=lat,
                                     lon2=data_frame.p[0],
                                     lat2=data_frame.p[1])

    distance_on_degree = abs(theta - data_frame.theta)
    if(distance_on_degree>180):
        distance_on_degree = distance_on_degree - 180

    thegma_2_on_position = -0.5*(max_distance_on_position**2)/math.log(final_discounted)
    thegma_2_on_degree = -0.5*(max_distance_on_degree**2)/math.log(final_discounted)

    '''guassion trustworthy transfer'''
    prob = 1.0 * math.exp(-1.0 / 2.0 * (distance_on_position**2) / (thegma_2_on_position)) * math.exp(-1.0 / 2.0 * (distance_on_degree**2) / (thegma_2_on_degree))

    return prob

def get_prob(lon, lat, theta, subjects, subjects_total, cur_data):

    prob_dic = []
    for i in range(subjects_total):
        x = get_transfered_data(lon=lon,
                                lat=lat,
                                theta=theta,
                                data_frame=subjects[i].data_frame[cur_data])
        prob_dic += [x]
    '''prob will not be normalized, since we do not encourage move to unmeasured area'''
    prob_sum = sum(prob_dic)

    prob_dic = np.array(prob_dic)

    '''normalize prob dic'''
    prob_dic_normalized = prob_dic / prob_sum

    distance_per_data = 0.0
    for i in range(subjects_total):
        distance_per_data += prob_dic_normalized[i] * subjects[i].data_frame[cur_data].v

    if(distance_per_data<=0):
        print("!!!! v too small !!!!")
        distance_per_data = 0.00001

    return (prob_sum/subjects_total), distance_per_data

def get_direction(lon, lat, direction, data_frame):

    '''
    Function: trustworthy transfer for theta domain
    Coder: syh
    Status: unchecked
    '''

    '''copy a data_frame to output'''
    data_frame_at_i = copy.copy(data_frame)

    sum_t = 0.0
    for i in range(360 / 8 * direction - 20, 360 / 8 * direction + 25, 360 / 8 / 9):

        sum_t += G_trustworth_transfer(lon = lon, lat = lat, data_frame = data_frame, theta = i)

    # sum_all = 0.0
    # for i in range(0, 360, 360 / 8 / 9):
    #
    #     sum_all += G_trustworth_transfer(lon = lon, lat = lat, data_frame = data_frame, theta = i)

    prob = sum_t

    return prob

def get_direction_all_expect(lon, lat, direction, subjects, data_frame_seq):

    '''
    Function: trustworthy transfer for theta domain
    Coder: syh
    Status: unchecked
    '''

    '''copy a data_frame to output'''
    direction_at = 0.0
    for i in range(len(subjects)):
        direction_at += get_direction(data_frame=subjects[i].data_frame[data_frame_seq],
                                     lon=lon,
                                     lat=lat,
                                     direction=direction)
    direction_at /= len(subjects)

    direction += 4
    if(direction>=8):
        direction -=8
    direction_at_op = 0.0
    for i in range(len(subjects)):
        direction_at_op += get_direction(data_frame=subjects[i].data_frame[data_frame_seq],
                                     lon=lon,
                                     lat=lat,
                                     direction=direction)
    direction_at_op /= len(subjects)

    # direction_at_all = 0.0
    # for d in range(8):
    #     direction_at_one = 0.0
    #     for i in range(len(subjects)):
    #         direction_at_one += get_direction(data_frame=subjects[i].data_frame[data_frame_seq],
    #                                          lon=lon,
    #                                          lat=lat,
    #                                          direction=d)
    #     direction_at_one /= len(subjects)
    # direction_at_all += direction_at_one


    return direction_at - direction_at_op # / direction_at_all

def calc_angle(x_point_s,y_point_s,x_point_e,y_point_e):

    '''
    Function: compute direction to north from Mercator cordication
    Coder: syh
    Status: unchecked
    '''

    angle=0
    y_se= y_point_e-y_point_s;
    x_se= x_point_e-x_point_s;
    if x_se==0 and y_se>0:
        angle = 360
    if x_se==0 and y_se<0:
        angle = 180
    if y_se==0 and x_se>0:
        angle = 90
    if y_se==0 and x_se<0:
        angle = 270
    if x_se>0 and y_se>0:
       angle = math.atan(x_se/y_se)*180/math.pi
    elif x_se<0 and y_se>0:
       angle = 360 + math.atan(x_se/y_se)*180/math.pi
    elif x_se<0 and y_se<0:
       angle = 180 + math.atan(x_se/y_se)*180/math.pi
    elif x_se>0 and y_se<0:
       angle = 180 + math.atan(x_se/y_se)*180/math.pi
    return angle

class subject():

    '''
    Function: data structure
    Coder: syh
    Status: coding
    '''

    def __init__(self, num_data_frame):
        self.data_frame = list(data_frame() for i in range(num_data_frame))

class data_frame():

    '''
    Function: data structure
    Coder: syh
    Status: coding
    '''

    def __init__(self):
        self.p = [0.0, 0.0] #position
        self.theta = 0.0 # direction
        self.v = 0.0 # speed
        self.t = 1.0 # trustworthy for position
        self.g = 1.0 # trustworthy for direction


def draw_direction(direction, data_frame_seq, subjects, draw_inter = 30):

    fig = plt.figure(1)
    plt.axis('normal')
    ax = Axes3D(fig)
    X = np.arange(-180, 180, draw_inter*2)
    Y = np.arange(-90, 90, draw_inter)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    x = 0
    for lon in range(-180, 180, draw_inter*2):
        y = 0
        for lat in range(-90, 90, draw_inter):
            direction_at = get_direction_all_expect( subjects=subjects,
                                                     lon=lon,
                                                     lat=lat,
                                                     direction=direction,
                                                     data_frame_seq=data_frame_seq)
            Z[x,y] = direction_at
            y += 1
        x += 1
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

def draw_directions(direction, data_frame_seq, subjects, draw_inter = 30):

    X = np.zeros((360/draw_inter, 180/draw_inter))
    Y = np.zeros((360/draw_inter, 180/draw_inter))
    sclaer = 0.5
    x=0
    for lon in range(-180, 180, draw_inter):
        y=0
        for lat in range(-90, 90, draw_inter):
            direction_at = get_direction_all_expect( subjects=subjects,
                                                     lon=lon,
                                                     lat=lat,
                                                     direction=direction,
                                                     data_frame_seq=data_frame_seq)
            # plt.arrow(lon, lat, direction_at*math.cos(direction*45), direction_at*math.sin(direction*45))
            X[x,y] = direction_at*math.cos(direction*45*math.pi/180)
            Y[x,y] = direction_at*math.sin(direction*45*math.pi/180)
            y += 1
        x += 1
    plt.figure(direction)
    plt.quiver(X, Y, color='#054E9F')

def avg_degree(angles):

    last_angle = angles[0];
    sum_angle = angles[0];
    for i in range(1, len(angles)):
        diff_angle = (angles[i] - angles[i-1] + 180) % (360) - 180;
        last_angle = last_angle + diff_angle;
        sum_angle = sum_angle + last_angle;
    avg = (sum_angle/len(angles)) % 360;
    return avg

def NDimensionGaussian(X_vector,U_Mean,CovarianceMatrix):
    #X=numpy.mat(X_vector)
    X=X_vector
    D=numpy.shape(X)[0]
    #U=numpy.mat(U_Mean)
    U=U_Mean
    #CM=numpy.mat(CovarianceMatrix)
    CM=CovarianceMatrix
    Y=X-U
    temp=Y.transpose() * CM.I * Y
    result=(1.0/((2*numpy.pi)**(D/2)))*(1.0/(numpy.linalg.det(CM)**0.5))*numpy.exp(-0.5*temp)
    return result

def CalMean(X):
    D,N=numpy.shape(X)
    MeanVector=numpy.mat(numpy.zeros((D,1)))
    for d in range(D):
        for n in range(N):
            MeanVector[d,0] += X[d,n]
        MeanVector[d,0] /= float(N)
    return MeanVector

def CalCovariance(X,MV):
    D,N=numpy.shape(X)
    CoV=numpy.mat(numpy.zeros((D,D)))
    for n in range(N):
        Temp=X[:,n]-MV
        CoV += Temp*Temp.transpose()
    CoV /= float(N)
    return CoV

def CalEnergy(Xn,Pik,Uk,Cov):
    D,N=numpy.shape(Xn)
    D_k,K=numpy.shape(Uk)
    if D!=D_k:
        print ('dimension not equal, break')
        return

    energy=0.0
    for n_iter in range(N):
        temp=0
        for k_iter in range(K):
            temp += Pik[0,k_iter] * NDimensionGaussian(Xn[:,n_iter],Uk[:,k_iter],Cov[k_iter])
        energy += numpy.log(temp)
    return float(energy)

def SequentialEMforMixGaussian(InputData,K):

    pi_Cof=numpy.mat(numpy.ones((1,K))*(1.0/float(K)))
    X=numpy.mat(InputData)
    X_mean=CalMean(X)
    print (X_mean)
    X_cov=CalCovariance(X,X_mean)
    print (X_cov)

    D,N=numpy.shape(X)
    print (D,N)
    UK=numpy.mat(numpy.zeros((D,K)))
    for d_iter in range(D):
        for k_iter in range(K):
            UK[d_iter,k_iter] = X_mean[d_iter,0] + (-1)**k_iter + (-1)**d_iter
    print (UK)

    List_cov=[]

    for k_iter in range(K):
        List_cov.append(numpy.mat(numpy.eye(X[:,0].size)))
    print (List_cov)

    List_cov_new=copy.deepcopy(List_cov)
    rZnk=numpy.mat(numpy.zeros((N,K)))
    denominator=numpy.mat(numpy.zeros((N,1)))
    rZnk_new=numpy.mat(numpy.zeros((N,K)))

    Nk=0.5*numpy.mat(numpy.ones((1,K)))
    print (Nk)
    Nk_new=numpy.mat(numpy.zeros((1,K)))
    UK_new=numpy.mat(numpy.zeros((D,K)))
    pi_Cof_new=numpy.mat(numpy.zeros((1,K)))

    for n_iter in range(1,N):
        #rZnk=pi_k*Gaussian(Xn|uk,Cov_k)/sum(pi_j*Gaussian(Xn|uj,Cov_j))
        for k_iter in range(K):
            rZnk_new[n_iter,k_iter] = pi_Cof[0,k_iter] * NDimensionGaussian(X[:,n_iter],UK[:,k_iter],List_cov[k_iter])
            denominator[n_iter,0] += rZnk_new[n_iter,k_iter]
        for k_iter in range(K):
            rZnk_new[n_iter,k_iter] /= denominator[n_iter,0]
            print ('rZnk_new', rZnk_new[n_iter,k_iter],'\n')
        for k_iter in range(K):
            Nk_new[0,k_iter] = Nk[0,k_iter] + rZnk_new[n_iter,k_iter] - rZnk[n_iter,k_iter]
            print ('Nk_new',Nk_new,'\n')

            pi_Cof_new[0,k_iter] = Nk_new[0,k_iter]/float(n_iter+1)
            print ('pi_Cof_new',pi_Cof_new,'\n')
            UK_new[:,k_iter] = UK[:,k_iter] + ( (rZnk_new[n_iter,k_iter] - rZnk[n_iter,k_iter])/float(Nk_new[0,k_iter]) ) * (X[:,n_iter]-UK[:,k_iter])
            print ('UK_new',UK_new,'\n')
            Temp = X[:,n_iter] - UK_new[:,k_iter]
            List_cov_new[k_iter] = List_cov[k_iter] + ((rZnk_new[n_iter,k_iter] - rZnk[n_iter,k_iter])/float(Nk_new[0,k_iter]))*(Temp*Temp.transpose()-List_cov[k_iter])
            print ('List_cov_new',List_cov_new,'\n')

        rZnk=copy.deepcopy(rZnk_new)
        pi_Cof=copy.deepcopy(pi_Cof_new)
        UK_new=copy.deepcopy(UK)
        List_cov=copy.deepcopy(List_cov_new)
    print (pi_Cof,UK_new,List_cov)
    return pi_Cof,UK_new,List_cov

def BatchEMforMixGaussian(InputData,K,MaxIter):

    pi_Cof=numpy.mat(numpy.ones((1,K))*(1.0/float(K)))
    X=numpy.mat(InputData)
    X_mean=CalMean(X)
    print (X_mean)
    X_cov=CalCovariance(X,X_mean)
    print (X_cov)

    D,N=numpy.shape(X)
    print (D,N)
    UK=numpy.mat(numpy.zeros((D,K)))
    for d_iter in range(D):
        for k_iter in range(K):
            UK[d_iter,k_iter] = X_mean[d_iter,0] + (-1)**k_iter + (-1)**d_iter
    print (UK)

    List_cov=[]

    for k_iter in range(K):
        List_cov.append(numpy.mat(numpy.eye(X[:,0].size)))
    print (List_cov)

    energy_new=0
    energy_old=CalEnergy(X,pi_Cof,UK,List_cov)
    print (energy_old)
    currentIter=0
    while True:
        currentIter += 1

        List_cov_new=[]
        rZnk=numpy.mat(numpy.zeros((N,K)))
        denominator=numpy.mat(numpy.zeros((N,1)))
        Nk=numpy.mat(numpy.zeros((1,K)))
        UK_new=numpy.mat(numpy.zeros((D,K)))
        pi_new=numpy.mat(numpy.zeros((1,K)))

        #rZnk=pi_k*Gaussian(Xn|uk,Cov_k)/sum(pi_j*Gaussian(Xn|uj,Cov_j))
        for n_iter in range(N):
            for k_iter in range(K):
                rZnk[n_iter,k_iter] = pi_Cof[0,k_iter] * NDimensionGaussian(X[:,n_iter],UK[:,k_iter],List_cov[k_iter])
                denominator[n_iter,0] += rZnk[n_iter,k_iter]
            for k_iter in range(K):
                rZnk[n_iter,k_iter] /= denominator[n_iter,0]
                #print 'rZnk', rZnk[n_iter,k_iter]

        #pi_new=sum(rZnk)
        for k_iter in range(K):
            for n_iter in range(N):
                Nk[0,k_iter] += rZnk[n_iter,k_iter]
            pi_new[0,k_iter] = Nk[0,k_iter]/(float(N))
            #print 'pi_k_new',pi_new[0,k_iter]

        #uk_new= (1/sum(rZnk))*sum(rZnk*Xn)
        for k_iter in range(K):
            for n_iter in range(N):
                UK_new[:,k_iter] += (1.0/float(Nk[0,k_iter]))*rZnk[n_iter,k_iter]*X[:,n_iter]
            #print 'UK_new',UK_new[:,k_iter]

        for k_iter in range(K):
            X_cov_new=numpy.mat(numpy.zeros((D,D)))
            for n_iter in range(N):
                Temp = X[:,n_iter] - UK_new[:,k_iter]
                X_cov_new += (1.0/float(Nk[0,k_iter]))*rZnk[n_iter,k_iter] * Temp * Temp.transpose()
            #print 'X_cov_new',X_cov_new
            List_cov_new.append(X_cov_new)

        energy_new=CalEnergy(X,pi_new,UK_new,List_cov)
        print ('energy_new',energy_new)
        #print pi_new
        #print UK_new
        #print List_cov_new
        if energy_old>=energy_new or currentIter>MaxIter:
            UK=copy.deepcopy(UK_new)
            pi_Cof=copy.deepcopy(pi_new)
            List_cov=copy.deepcopy(List_cov_new)
            break
        else:
            UK=copy.deepcopy(UK_new)
            pi_Cof=copy.deepcopy(pi_new)
            List_cov=copy.deepcopy(List_cov_new)
            energy_old=energy_new

    return pi_Cof,UK,List_cov


def compute_consi(data):

    '''get essiancial config'''
    num_subject = np.shape(data)[1] / 2
    num_data_frame = np.shape(data)[0]

    '''subjects'''
    subjects = list(subject(num_data_frame=num_data_frame) for i in range(num_subject))

    '''set in data from mat to structure'''
    for subject_i in range(num_subject):

        for data_frame_i in range(num_data_frame):

            subjects[subject_i].data_frame[data_frame_i].p[0] = data[data_frame_i, subject_i*2 + 1] #lon
            subjects[subject_i].data_frame[data_frame_i].p[1] = data[data_frame_i, subject_i*2] #lat

        for data_frame_i in range(1, num_data_frame - 1):

            last_frame = subjects[subject_i].data_frame[data_frame_i - 1]
            next_frame = subjects[subject_i].data_frame[data_frame_i + 1]
            v = haversine(lon1=last_frame.p[0],
                          lat1=last_frame.p[1],
                          lon2=next_frame.p[0],
                          lat2=next_frame.p[1])
            l_x, l_y = lonlat2Mercator(lon=last_frame.p[0],
                                       lat=last_frame.p[1])
            n_x, n_y = lonlat2Mercator(lon=next_frame.p[0],
                                       lat=next_frame.p[1])
            theta = calc_angle(x_point_s=l_x,
                               y_point_s=l_y,
                               x_point_e=n_x,
                               y_point_e=n_y)
            subjects[subject_i].data_frame[data_frame_i].v = v
            if(v>(no_moving_gate)):
                subjects[subject_i].data_frame[data_frame_i].theta = theta
            else:
                subjects[subject_i].data_frame[data_frame_i].theta = 'null'
            subjects[subject_i].data_frame[data_frame_i].t = 1.0


    sum_on_video = np.zeros((NumDirectionForCluster))
    count_on_video = 0
    valid_circle_count = 0
    for data_frame_i in range(0 + frame_gate, num_data_frame - frame_gate - 1):
        sum_on_frame = np.zeros((NumDirectionForCluster))
        count_on_frame=0
        valid_circle_count_last = valid_circle_count
        for lon_i in range(-180, 180, compute_lon_inter):
            for lat_i in range(-90, 90, compute_lat_inter):

                theta_dic = []
                for data_frame_i_in_i in range(data_frame_i - frame_gate, data_frame_i + frame_gate + 1):
                    for subject_i in range(num_subject):
                        distance = haversine(lon1=lon_i,
                                             lat1=lat_i,
                                             lon2=subjects[subject_i].data_frame[data_frame_i_in_i].p[0],
                                             lat2=subjects[subject_i].data_frame[data_frame_i_in_i].p[1])
                        if(distance<1.0*(fov_degree*math.pi/180.0)):
                            if(subjects[subject_i].data_frame[data_frame_i_in_i].theta != 'null'):
                                theta_dic += [subjects[subject_i].data_frame[data_frame_i_in_i].theta]

                if(len(theta_dic)>=(NumDirectionForCluster)):
                    direction_dic = np.zeros((NumDirectionForCluster))
                    for i in range(NumDirectionForCluster):
                        direction_dic[i] = 0
                    detect = False
                    for theta_i in range(len(theta_dic)):
                        if((theta_dic[theta_i]>(360.0-DirectionInter/2.0)) or (theta_dic[theta_i]<=(0.0+DirectionInter/2.0))):
                            direction_dic[0] += 1
                            detect = True
                        else:
                            for direction_i in range(1, NumDirectionForCluster):
                                if((direction_i*DirectionInter - DirectionInter/2.0)<theta_dic[theta_i]<=(direction_i*DirectionInter + DirectionInter/2.0)):
                                    direction_dic[direction_i] += 1
                                    detect = True
                        if(detect==False):
                            print('!!!!')
                    # print(direction_dic)
                    direction_dic = np.sort(direction_dic)
                    # print(direction_dic)
                    direction_dic = direction_dic / np.sum(direction_dic)

                    sum_on_frame += direction_dic
                    count_on_frame += 1
                    valid_circle_count += 1


        cov_on_frame = sum_on_frame / count_on_frame
        sum_on_video += cov_on_frame
        count_on_video += 1
        valid_circle_count_thisframe = valid_circle_count - valid_circle_count_last
        display_string = 'frame\t'+str(data_frame_i)+'\tvalid_circle\t'+str(valid_circle_count_thisframe)
        for print_i in range(NumDirectionForCluster):
            display_string += ('\t\t' + str(cov_on_frame[print_i]))
        print(display_string)
    cov_on_video = sum_on_video / count_on_video
    valid_circle_exp_per_frame = valid_circle_count * 1.0 / count_on_frame

    return cov_on_video, valid_circle_exp_per_frame

def get_num_subjects(data):
    num_subject = np.shape(data)[1] / 2
    return num_subject

def get_subjects(data, subject_id):

    '''get essiancial config'''
    num_subject = np.shape(data)[1] / 2
    num_data_frame = np.shape(data)[0]

    '''subjects'''
    subjects = list(subject(num_data_frame=num_data_frame) for i in range(num_subject))

    '''set in data from mat to structure'''
    for subject_i in range(num_subject):

        for data_frame_i in range(num_data_frame):

            subjects[subject_i].data_frame[data_frame_i].p[0] = data[data_frame_i, subject_i*2 + 1] #lon
            subjects[subject_i].data_frame[data_frame_i].p[1] = data[data_frame_i, subject_i*2] #lat

        for data_frame_i in range(1, num_data_frame - 1):

            last_frame = subjects[subject_i].data_frame[data_frame_i - 1]
            next_frame = subjects[subject_i].data_frame[data_frame_i + 1]
            v = haversine(lon1=last_frame.p[0],
                          lat1=last_frame.p[1],
                          lon2=next_frame.p[0],
                          lat2=next_frame.p[1]) / 2.0
            l_x, l_y = lonlat2Mercator(lon=last_frame.p[0],
                                       lat=last_frame.p[1])
            n_x, n_y = lonlat2Mercator(lon=next_frame.p[0],
                                       lat=next_frame.p[1])
            theta = calc_angle(x_point_s=l_x,
                               y_point_s=l_y,
                               x_point_e=n_x,
                               y_point_e=n_y)

            subjects[subject_i].data_frame[data_frame_i].v = v
            subjects[subject_i].data_frame[data_frame_i].theta = theta

            if(subject_i==1):
                subjects[subject_i].data_frame[0].v = v
                subjects[subject_i].data_frame[0].theta = theta
            if(subject_i==(num_data_frame - 2)):
                subjects[subject_i].data_frame[num_data_frame - 1].v = v
                subjects[subject_i].data_frame[num_data_frame - 1].theta = theta

    return num_subject, num_data_frame, subjects, subjects[subject_id]

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

        print(observation)
        print(np.shape(observation))
        print(reward)
        print(done)
        print(to_log)
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
