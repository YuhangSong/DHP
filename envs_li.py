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

final_discount = 4
final_discounted_config = 10**(-final_discount)
v_relative_discounted_config = 0.5

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
        self.temp_dir = "temp/get_view/w_" + str(self.worker_id) + "/"
        '''clear temp dir for this worker'''
        subprocess.call(["rm", "-r", self.temp_dir])
        subprocess.call(["mkdir", "-p", self.temp_dir])

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
        self.observation = self.reset()

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

    def reset(self):

        '''reset cur_step and cur_data'''
        self.cur_step = 0
        self.cur_data = 0

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
            self.reset()
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
            reward = last_prob
            done = False

        return self.cur_observation, reward, done
