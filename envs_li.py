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
from suppor_lib import *
from move_view_lib_new import view_mover

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()

'''
Warning: all degree in du
         lon from -180 to 180
'''

class env_li():

    '''
    Function: env interface for ff
    Coder: syh
    Status: checking
    '''

    def __init__(self, env_id, task):

        '''only log if the task is on zero and cluster is the main cluster'''
        self.task = task

        '''get id contains only name of the video'''
        self.env_id = env_id

        from config import reward_estimator
        self.reward_estimator = reward_estimator

        '''load config'''
        self.config()

        '''create view_mover'''
        self.view_mover = view_mover()

        '''reset'''
        self.observation = self.reset()

    def get_observation(self):

        '''interface to get view'''
        self.cur_observation = get_view(input_width=self.video_size_width,
                                        input_height=self.video_size_heigth,
                                        view_fov_x=self.view_range_lon,
                                        view_fov_y=self.view_range_lat,
                                        cur_frame=self.cur_frame,
                                        is_render=False,
                                        output_width=np.shape(self.observation_space)[0],
                                        output_height=np.shape(self.observation_space)[1],
                                        view_center_lon=self.cur_lon,
                                        view_center_lat=self.cur_lat,
                                        temp_dir=self.temp_dir,
                                        file_='../../vr/' + self.env_id + '.yuv')

    def config(self):

        '''function to load config'''
        print("=================config=================")

        '''observation_space'''
        from config import observation_space
        self.observation_space = observation_space

        '''set all temp dir for this worker'''
        self.temp_dir = "temp/get_view/w_" + str(self.task) + '/'
        print(self.task)
        print(self.temp_dir)
        '''clear temp dir for this worker'''
        subprocess.call(["rm", "-r", self.temp_dir])
        subprocess.call(["mkdir", "-p", self.temp_dir])

        '''load in mat data of head movement'''
        matfn = '../../vr/FULLdata_per_video_frame.mat'
        data_all = sio.loadmat(matfn)
        data = data_all[self.env_id]
        self.subjects_total = get_num_subjects(data=data)

        print("env set to: "+str(self.env_id))

        '''frame bug'''
        '''some bug in the frame read for some video,='''
        if(self.env_id=='Dubai'):
            self.frame_bug_offset = 540
        elif(self.env_id=='MercedesBenz'):
            self.frame_bug_offset = 10
        elif(self.env_id=='Cryogenian'):
            self.frame_bug_offset = 10
        else:
            self.frame_bug_offset = 0

        '''get subjects'''
        self.subjects_total, self.data_total, self.subjects, _ = get_subjects(data,0)

        '''init video and get paramters'''
        video = cv2.VideoCapture('../../vr/' + self.env_id + '.mp4')
        self.frame_per_second = video.get(cv2.cv.CV_CAP_PROP_FPS)
        self.frame_total = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        self.video_size_width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        self.video_size_heigth = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        self.second_total = self.frame_total / self.frame_per_second
        self.data_per_frame = self.data_total / self.frame_total

        '''compute step lenth from data_tensity'''
        from config import data_tensity
        self.second_per_step = max(data_tensity/self.frame_per_second, data_tensity/self.data_per_frame/self.frame_per_second)
        self.frame_per_step = self.frame_per_second * self.second_per_step
        self.data_per_step = self.data_per_frame * self.frame_per_step

        '''compute step_total'''
        self.step_total = int(self.data_total / self.data_per_step) + 1

        '''set fov range'''
        from config import view_range_lon, view_range_lat
        self.view_range_lon = view_range_lon
        self.view_range_lat = view_range_lat

        self.episode = 0

        '''salmap'''
        self.heatmap_height = 180
        self.heatmap_width = 360

        '''load ground-truth heat map'''
        from config import heatmap_sigma
        gt_heatmap_dir = 'gt_heatmap_sp_' + heatmap_sigma
        self.gt_heatmaps = self.load_heatmaps(gt_heatmap_dir)

        from config import num_workers_global,cluster_current,cluster_main
        if (self.task%num_workers_global==0) and (cluster_current==cluster_main):
            print('>>>>>>>>>>>>>>>>>>>>this is a log thread<<<<<<<<<<<<<<<<<<<<<<<<<<')
            self.log_thread = True
        else:
            self.log_thread = False

        '''update settings for log_thread'''
        if self.log_thread:
            self.log_thread_config()

    def log_thread_config(self):

        from config import if_log_scan_path
        self.if_log_scan_path = if_log_scan_path

        from config import if_log_cc
        self.if_log_cc = if_log_cc

        if self.if_log_cc:
            '''cc record'''
            self.agent_result_saver = []
            self.agent_result_stack = []

            self.max_cc = 0.0
            self.cur_cc = 0.0

            from config import relative_predicted_fixation_num
            self.predicted_fixtions_num = int(self.subjects_total * relative_predicted_fixation_num)
            print('predicted_fixtions_num is '+str(self.predicted_fixtions_num))
            from config import relative_log_cc_interval
            self.if_log_cc_interval = int(self.predicted_fixtions_num * relative_log_cc_interval)
            print('log_cc_interval is '+str(self.if_log_cc_interval))

    def reset(self):

        '''reset cur_step and cur_data'''
        self.cur_step = 0
        self.cur_data = 0

        '''episode add'''
        self.episode +=1

        '''reset cur_frame'''
        self.cur_frame = 0

        '''reset cur_lon and cur_lat to one of the subjects start point'''
        subject_dic_code = []
        for i in range(self.subjects_total):
            subject_dic_code += [i]
        subject_code = np.random.choice(a=subject_dic_code)
        self.cur_lon = self.subjects[subject_code].data_frame[0].p[0]
        self.cur_lat = self.subjects[subject_code].data_frame[0].p[1]

        '''reset view_mover'''
        self.view_mover.init_position(Latitude=self.cur_lat,
                                      Longitude=self.cur_lon)

        '''set observation_now to the first frame'''
        self.get_observation()

        self.last_observation = None

        if self.log_thread:
            self.log_thread_reset()

        return self.cur_observation

    def log_thread_reset(self):

        if self.if_log_scan_path:
            plt.figure(str(self.env_id)+'_scan_path')
            plt.clf()

        if self.if_log_cc:

            self.agent_result_stack += [copy.deepcopy(self.agent_result_saver)]
            self.agent_result_saver = []

            if len(self.agent_result_stack) > self.predicted_fixtions_num:

                '''if stack full, pop out the oldest data'''
                self.agent_result_stack.pop(0)

                if self.episode%self.if_log_cc_interval is 0:

                    print('compute cc..................')

                    ccs_on_step_i = []
                    heatmaps_on_step_i = []
                    for step_i in range(self.step_total):

                        '''generate predicted salmap'''
                        temp = np.asarray(self.agent_result_stack)[:,step_i]
                        temp = np.sum(temp,axis=0)
                        temp = temp / np.max(temp)
                        heatmaps_on_step_i += [copy.deepcopy(temp)]
                        from cc import calc_score
                        ccs_on_step_i += [copy.deepcopy(calc_score(self.gt_heatmaps[step_i], heatmaps_on_step_i[step_i]))]
                        print('cc on step '+str(step_i)+' is '+str(ccs_on_step_i[step_i]))

                    self.cur_cc = np.mean(np.asarray(ccs_on_step_i))
                    print('cur_cc is '+str(self.cur_cc))
                    if self.cur_cc > self.max_cc:
                        print('new max cc found: '+str(self.cur_cc)+', recording cc and heatmaps')
                        self.max_cc = self.cur_cc
                        self.heatmaps_of_max_cc = heatmaps_on_step_i

                        from config import final_log_dir
                        record_dir = final_log_dir+'ff_best_heatmaps/'+self.env_id+'/'
                        subprocess.call(["rm", "-r", record_dir])
                        subprocess.call(["mkdir", "-p", record_dir])
                        for step_i in range(self.step_total):
                            self.save_heatmap(heatmap=self.heatmaps_of_max_cc[step_i],
                                              path=record_dir,
                                              name=str(step_i))

    def step(self, action):

        '''these will be returned, but not sure to updated'''
        if self.log_thread:
            self.log_thread_step()

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
            self.cur_lon, self.cur_lat = self.view_mover.move_view(direction=action * 45.0,degree_per_step=degree_per_step)

            '''update observation_now'''
            self.get_observation()

            '''produce output'''
            if self.reward_estimator is 'trustworthy_transfer':
                reward = last_prob
            elif self.reward_estimator is 'cc':
                cur_heatmap = fixation2salmap(fixation=[[self.cur_lon, self.cur_lat]],
                                              mapwidth=self.heatmap_width,
                                              mapheight=self.heatmap_height)
                from cc import calc_score
                reward = calc_score(self.gt_heatmaps[self.cur_step], cur_heatmap)
            done = False

        if self.log_thread:
            if self.if_log_cc:
                return self.cur_observation, reward, done, self.cur_cc, self.max_cc
        return self.cur_observation, reward, done, 0.0, 0.0

    def log_thread_step(self):
        '''log_scan_path'''
        if self.if_log_scan_path:
            plt.figure(str(self.env_id)+'_scan_path')
            plt.scatter(self.cur_lon, self.cur_lat, c='r')
            plt.scatter(-180, -90)
            plt.scatter(-180, 90)
            plt.scatter(180, -90)
            plt.scatter(180, 90)
            plt.pause(0.00001)

        if self.if_log_cc:
            self.agent_result_saver += [copy.deepcopy(fixation2salmap(fixation=[[self.cur_lon,self.cur_lon]],
                                                                      mapwidth=self.heatmap_width,
                                                                      mapheight=self.heatmap_height))]

    def load_heatmaps(self, name):

        heatmaps = []
        for step in range(self.step_total):

            try:
                file_name = '../../vr/'+name+'/'+self.env_id+'_'+str(step)+'.jpg'
                temp = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                temp = cv2.resize(temp,(self.heatmap_width, self.heatmap_height))
                temp = temp / 255.0
                heatmaps += [temp]
            except Exception,e:
                print Exception,":",e
                continue

        print('load heatmaps: '+name+' done, size: '+str(np.shape(heatmaps)))

        return heatmaps

    def save_heatmap(self,heatmap,path,name):
        heatmap = heatmap * 255.0
        cv2.imwrite(path+'/'+name+'.jpg',heatmap)
