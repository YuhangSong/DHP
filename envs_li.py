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
import tensorflow as tf

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

    def __init__(self, env_id, task, subject=None, summary_writer=None):

        '''only log if the task is on zero and cluster is the main cluster'''
        self.task = task

        ''''''
        self.summary_writer = summary_writer

        '''get id contains only name of the video'''
        self.env_id = env_id
        from config import game_dic
        self.env_id_num = game_dic.index(self.env_id)

        from config import reward_estimator
        self.reward_estimator = reward_estimator

        from config import mode
        self.mode = mode

        self.subject = subject

        '''load config'''
        self.config()

        '''create view_mover'''
        from config import use_move_view_lib
        self.use_move_view_lib = use_move_view_lib
        if self.use_move_view_lib is 'new':
            from move_view_lib_new import view_mover
            self.view_mover = view_mover()

        '''reset'''
        self.observation = self.reset()

        # self.terminate_this_worker()

        # self.max_cc = self.env_id_num
        # self.write_best_cc()
        # print(s)

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
                                        file_='../../'+self.data_base+'/' + self.env_id + '.yuv')

    def config(self):

        '''function to load config'''
        print("=================config=================")

        from config import data_base
        self.data_base = data_base

        if self.mode is 'on_line':
            from config import if_run_baseline
            self.if_run_baseline = if_run_baseline
            if self.if_run_baseline is True:
                from config import baseline_type, v_used_in_baseline
                self.baseline_type = baseline_type
                self.v_used_in_baseline = v_used_in_baseline

        from config import if_learning_v
        self.if_learning_v = if_learning_v

        '''observation_space'''
        from config import observation_space
        self.observation_space = observation_space

        '''set all temp dir for this worker'''
        if (self.mode is 'off_line') or (self.mode is 'data_processor'):
            self.temp_dir = "temp/get_view/w_" + str(self.task) + '/'
        elif self.mode is 'on_line':
            self.temp_dir = "temp/get_view/g_" + str(self.env_id) + '_s_' + str(self.subject) + '/'
        print(self.task)
        print(self.temp_dir)
        '''clear temp dir for this worker'''
        subprocess.call(["rm", "-r", self.temp_dir])
        subprocess.call(["mkdir", "-p", self.temp_dir])

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
        '''load in mat data of head movement'''
        matfn = '../../'+self.data_base+'/FULLdata_per_video_frame.mat'
        data_all = sio.loadmat(matfn)
        data = data_all[self.env_id]
        self.subjects_total, self.data_total, self.subjects, _ = get_subjects(data,0)

        self.reward_dic_on_cur_episode = []

        if self.mode is 'on_line':
            self.subjects_total = 1
            self.subjects = self.subjects[self.subject:self.subject+1]
            self.cur_training_step = 0.0
            self.cur_predicting_step = self.cur_training_step + 1.0
            self.predicting = False
            from config import train_to_reward, train_to_mo
            self.train_to_reward = train_to_reward
            self.train_to_mo = train_to_mo
            from config import train_to_episode
            self.train_to_episode = train_to_episode
            self.sum_reward_dic_on_cur_train = []
            self.average_reward_dic_on_cur_train = []

            '''record mo'''
            self.mo_dic_on_cur_episode = []
            self.sum_mo_dic_on_cur_train = []
            self.average_mo_dic_on_cur_train = []
            self.mo_on_prediction_dic = []

        '''init video and get paramters'''
        video = cv2.VideoCapture('../../'+self.data_base+'/' + self.env_id + '.mp4')
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

        self.max_cc = 0.0
        self.cur_cc = 0.0

        '''salmap'''
        self.heatmap_height = 180
        self.heatmap_width = 360

        if self.mode is 'data_processor':
            self.data_processor()

        '''load ground-truth heat map'''
        from config import heatmap_sigma
        gt_heatmap_dir = 'gt_heatmap_sp_' + heatmap_sigma
        self.gt_heatmaps = self.load_heatmaps(gt_heatmap_dir)

        if (self.mode is 'off_line') or (self.mode is 'data_processor'):
            if (self.task==0):
                print('>>>>>>>>>>>>>>>>>>>>this is a log thread<<<<<<<<<<<<<<<<<<<<<<<<<<')
                self.log_thread = True
            else:
                self.log_thread = False
        elif self.mode is 'on_line':
            print('>>>>>>>>>>>>>>>>>>>>this is a log thread<<<<<<<<<<<<<<<<<<<<<<<<<<')
            self.log_thread = True

        '''update settings for log_thread'''
        if self.log_thread:
            self.log_thread_config()

    def data_processor(self):
        from config import data_processor_id
        print('==========================data process start: '+data_processor_id+'================================')
        if data_processor_id is 'minglang_mp4_to_yuv':
            print('sssss')
            from config import game_dic_new_all
            for i in range(len(game_dic_new_all)):
                # print(game_dic_new_all[i])
                if i >= 0 and i <= 0: #len(game_dic_new_all)
                    # file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+str(game_dic_new_all[i])+'.mp4'
                    file_out_1 = '/media/minglang/YuhangSong_1/ff/vr_yuv/'+"Let'sNotBeAloneTonight"+'.yuv'
                    file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+"Let'sNotBeAloneTonight"+'.mp4'
                    self.video = cv2.VideoCapture(file_in_1)
                    input_width_1 = int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
                    input_height_1 = int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
                    self.mp4_to_yuv(input_width_1,input_height_1,file_in_1,file_out_1)
                    print('end processing: ',file_out_1)

            # print('len_game_dic_new_all: ',len(game_dic_new_all))
            # print('get_view')

            # print(game_dic_new_all)

        if data_processor_id is 'minglang_mp4_to_jpg':
            from config import game_dic_new_all
            for i in range(len(game_dic_new_all)):
                # print(game_dic_new_all[i])
                if i >= 1 and i <= len(game_dic_new_all): #len(game_dic_new_all)
                    # file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+str(game_dic_new_all[i])+'.mp4'
                    # file_out_1 = '/media/minglang/YuhangSong_1/ff/vr_yuv/'+"Let'sNotBeAloneTonight"+'.yuv'
                    # file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+"Let'sNotBeAloneTonight"+'.mp4'
                    file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+str(game_dic_new_all[i])+'.mp4'
                    file_out_1 = '/media/minglang/YuhangSong_1/ff/vr_bms_jpg/'+str(game_dic_new_all[i])+'.yuv'

                    video = cv2.VideoCapture(file_in_1)
                    self.video = video
                    self.frame_per_second = round(video.get(cv2.cv.CV_CAP_PROP_FPS))
                    self.frame_total = round(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

                    for frame_i in range(int(self.frame_total)):

                        try:
                            rval, frame = self.video.read()
                            # here minglang 1
                            cv2.imwrite('/media/minglang/YuhangSong_1/ff/vr_bms_jpg/'+str(game_dic_new_all[i])+'_'+str(frame_i)+'.jpg',frame)
                            print(frame_i)
                        except Exception, e:
                            print('failed on this frame, continue')
                            print Exception,":",e
                            continue

                    print('end processing: ',file_in_1,self.frame_per_second,self.frame_total)

        if data_processor_id is 'minglang_obdl_cfg':
            from config import game_dic_new_all
            for i in range(len(game_dic_new_all)):
                # print(game_dic_new_all[i])

                if i >= 100 and i <=  len(game_dic_new_all): #len(game_dic_new_all)
                    # file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+str(game_dic_new_all[i])+'.mp4'
                    # file_out_1 = '/media/minglang/YuhangSong_1/ff/vr_yuv/'+"Let'sNotBeAloneTonight"+'.yuv'
                    # file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+"Let'sNotBeAloneTonight"+'.mp4'
                    file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+str(game_dic_new_all[i])+'.mp4'
                    CONFIG_FILE = '/media/minglang/YuhangSong_1/ff/obdl_vr_new/'+str(game_dic_new_all[i])+'.cfg'

                    # # get the paramters
                    video = cv2.VideoCapture(file_in_1)
                    self.video = video
                    self.frame_per_second = int(round(video.get(cv2.cv.CV_CAP_PROP_FPS)))
                    self.frame_total = int(round(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)))
                    NAME = game_dic_new_all[i]
                    FRAMESCOUNT = self.frame_total
                    FRAMERATE = self.frame_per_second
                    IMAGEWIDTH = int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
                    IMAGEHEIGHT = int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

                    # write the paramters throuh cfg
                    # conf = ConfigParser.ConfigParser()
                    # cfgfile = open(CONFIG_FILE,'w')
                    # # conf.add_section("")

                    # write through txt
                    f_config = open(CONFIG_FILE,"w")
                    f_config.write("NAME\n")
                    f_config.write(str(game_dic_new_all[i])+'\n')
                    f_config.write("FRAMESCOUNT\n")
                    f_config.write(str(FRAMESCOUNT)+'\n')
                    f_config.write("FRAMERATE\n")
                    f_config.write(str(FRAMERATE)+'\n')
                    f_config.write("IMAGEWIDTH\n")
                    f_config.write(str(IMAGEWIDTH)+'\n')
                    f_config.write("IMAGEHEIGHT\n")
                    f_config.write(str(IMAGEHEIGHT)+'\n')
                    f_config.close()

                #one video and one cfg in one file
                if i >= 0 and i <= len(game_dic_new_all): #len(game_dic_new_all)
                    cfg_file = '/media/minglang/YuhangSong_1/ff/obdl_vr_new/obdl_vr_new/'+str(game_dic_new_all[i])
                    os.makedirs(cfg_file)
                    file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+str(game_dic_new_all[i])+'.mp4'
                    shutil.copy(file_in_1,cfg_file)
                    CONFIG_FILE = '/media/minglang/YuhangSong_1/ff/obdl_vr_new/'+str(game_dic_new_all[i])+'.cfg'
                    shutil.copy(CONFIG_FILE,cfg_file)
                    print("os.makedirs(cfg_file)")


        print('=============================data process end, programe terminate=============================')
        print(t)

    def log_thread_config(self):

        from config import if_log_scan_path
        self.if_log_scan_path = if_log_scan_path

        from config import if_log_cc
        self.if_log_cc = if_log_cc

        if self.if_log_cc:

            if self.mode is 'off_line':
                '''cc record'''
                self.agent_result_saver = []
                self.agent_result_stack = []


                if self.mode is 'off_line': # yuhangsong here
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

        self.reward_dic_on_cur_episode = []

        if self.mode is 'on_line':
            self.mo_dic_on_cur_episode = []

        '''episode add'''
        self.episode +=1

        '''reset cur_frame'''
        self.cur_frame = 0

        '''reset last action'''
        self.last_action = None

        '''reset cur_lon and cur_lat to one of the subjects start point'''
        subject_dic_code = []
        for i in range(self.subjects_total):
            subject_dic_code += [i]
        if self.mode is 'off_line':
            subject_code = np.random.choice(a=subject_dic_code)
        elif self.mode is 'on_line':
            subject_code = 0
        self.cur_lon = self.subjects[subject_code].data_frame[0].p[0]
        self.cur_lat = self.subjects[subject_code].data_frame[0].p[1]

        '''reset view_mover'''
        if self.use_move_view_lib is 'new':
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

            if self.mode is 'off_line':

                self.agent_result_stack += [copy.deepcopy(self.agent_result_saver)]
                self.agent_result_saver = []

                if len(self.agent_result_stack) > self.predicted_fixtions_num:

                    '''if stack full, pop out the oldest data'''
                    self.agent_result_stack.pop(0)

                    if self.episode%self.if_log_cc_interval is 0:

                        print('compute cc..................')

                        ccs_on_step_i = []
                        heatmaps_on_step_i = []
                        for step_i in range(self.step_total-1):

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

                            '''log'''
                            from config import final_log_dir
                            record_dir = final_log_dir+'ff_best_heatmaps/'+self.env_id+'/'
                            subprocess.call(["rm", "-r", record_dir])
                            subprocess.call(["mkdir", "-p", record_dir])
                            for step_i in range(self.step_total-1):
                                self.save_heatmap(heatmap=self.heatmaps_of_max_cc[step_i],
                                                  path=record_dir,
                                                  name=str(step_i))

                            self.write_best_cc()

    def write_best_cc(self):
        from config import final_log_dir
        record_dir = final_log_dir+'ff_best_cc/'+self.env_id+'/'
        while True:
            try:
                subprocess.call(["rm", "-r", record_dir])
                subprocess.call(["mkdir", "-p", record_dir])
                np.savez(record_dir+'best_cc.npz',
                         best_cc=[self.max_cc])
                break
            except Exception, e:
                print(str(Exception)+": "+str(e))
                time.sleep(1)

    def step(self, action, v):

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

        v_lable = 0.0

        '''if any of update frame or update data is failed'''
        if(update_frame_success==False)or(update_data_success==False):

            '''terminating'''
            self.reset()
            reward = 0.0
            mo = 0.0
            done = True
            if self.if_learning_v:
                v_lable = 0.0

        else:

            if self.mode is 'on_line':

                if self.if_run_baseline is True:

                    '''if run baseline, overwrite the action and v'''


                    if self.baseline_type is 'keep':
                        from suppor_lib import constrain_degree_to_0_360
                        action = int(round((constrain_degree_to_0_360(self.subjects[0].data_frame[self.last_data].theta))/45.0)) # constrain to 0~360, /45.0 round
                    elif self.baseline_type is 'random':
                        import random
                        action = random.randint(0,7)


                    '''overwrite v ,if v<0,action turn to the opposite'''
                    self.v_expectation_used_in_baseline = self.v_used_in_baseline * self.data_per_step / math.pi *180.0
                    self.v_stdev_used_in_baseline = self.v_expectation_used_in_baseline
                    v = numpy.random.normal(self.v_expectation_used_in_baseline,self.v_stdev_used_in_baseline)
                    if v < 0:
                        action = (action + 4) % 8
                        v = 0 - v

            '''get direction reward and ground-truth v from data_base in last state'''
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
            '''set v_lable'''
            v_lable = degree_per_step

            '''move view, update cur_lon and cur_lat, the standard procedure of rl'''
            if self.if_learning_v:
                v_used_to_step = v
            else:
                v_used_to_step = v_lable

            if self.use_move_view_lib is 'new':
                self.cur_lon, self.cur_lat = self.view_mover.move_view(direction=action * 45.0,
                                                                       degree_per_step=v_used_to_step)
            elif self.use_move_view_lib is 'ziyu':
                from move_view_lib import move_view
                self.cur_lon, self.cur_lat = move_view(cur_lon=self.last_lon,
                                                       cur_lat=self.last_lat,
                                                       direction=action,
                                                       degree_per_step=v_used_to_step)
            self.last_action = action

            '''produce reward'''
            if self.reward_estimator is 'trustworthy_transfer':
                reward = last_prob
            elif self.reward_estimator is 'cc':
                cur_heatmap = fixation2salmap(fixation=[[self.cur_lon, self.cur_lat]],
                                              mapwidth=self.heatmap_width,
                                              mapheight=self.heatmap_height)
                from cc import calc_score
                reward = calc_score(self.gt_heatmaps[self.cur_step], cur_heatmap)

            if self.mode is 'on_line':

                '''compute MO'''
                from MeanOverlap import *
                mo_calculator = MeanOverlap(self.video_size_width,
                                            self.video_size_heigth,
                                            65.5/2,
                                            3.0/4.0)
                mo = mo_calculator.calc_mo_deg((self.cur_lon,self.cur_lat),(self.subjects[0].data_frame[self.cur_data].p[0],self.subjects[0].data_frame[self.cur_data].p[1]),is_centered = True)
                self.mo_dic_on_cur_episode += [mo]

            '''smooth reward'''
            if self.last_action is not None:

                '''if we have last_action'''

                '''compute smooth reward'''
                action_difference = abs(action-self.last_action)
                from config import direction_num
                if action_difference > (direction_num/2):
                    action_difference -= (direction_num/2)
                from config import reward_smooth_discount_to
                reward *= (1.0-(action_difference*(1.0-reward_smooth_discount_to)/(direction_num/2)))

            '''record'''
            self.reward_dic_on_cur_episode += [reward]


            '''
                All reward and scores has been computed, we now consider if we want to drawback the position
            '''
            if (self.mode is 'on_line'):

                if self.if_run_baseline is True:

                    '''
                        if run baseline, should draw back
                    '''
                    print('>>>>>>Draw position back>>>>>>>')
                    self.cur_lon = self.subjects[0].data_frame[self.cur_data].p[0]
                    self.cur_lat = self.subjects[0].data_frame[self.cur_data].p[1]

                if (self.predicting is True) or (self.if_run_baseline is True):

                    '''
                        if we are predicting we are actually feeding the model so that we can produce
                        a prediction with the experiences already experienced by the human.
                    '''
                    print('>>>>>>Draw position back>>>>>>>')
                    self.cur_lon = self.subjects[0].data_frame[self.cur_data].p[0]
                    self.cur_lat = self.subjects[0].data_frame[self.cur_data].p[1]

            '''
                after pull the position, get observation
                update observation_now
            '''
            self.get_observation()

            '''
                normally, we donot judge done when we in this
            '''
            done = False

            '''
                core part for online
            '''
            if self.mode is 'on_line':

                if (self.if_run_baseline is True):

                    '''if running baseline, we are always predicting'''
                    self.predicting = True

                    '''we predict until the last step'''
                    self.cur_predicting_step = self.step_total - 4


                if self.predicting is False:

                    '''if is training'''
                    if self.cur_step > self.cur_training_step:

                        '''if step is out of training range'''

                        if (np.mean(self.reward_dic_on_cur_episode) > self.train_to_reward) or (np.mean(self.mo_dic_on_cur_episode) > self.train_to_mo) or (len(self.sum_reward_dic_on_cur_train)>self.train_to_episode):

                            '''if reward is trained to a acceptable range or trained episode exceed a range'''
                            '''or is running baseline'''

                            print('>>>>>train to an acceptable state')

                            '''summary'''
                            summary = tf.Summary()
                            '''summary reward'''
                            summary.value.add(tag=self.env_id+'on_cur_train/number_of_episodes',
                                              simple_value=float(len(self.sum_reward_dic_on_cur_train)))
                            summary.value.add(tag=self.env_id+'on_cur_train/average_@sum_reward_per_step@',
                                              simple_value=float(np.mean(self.sum_reward_dic_on_cur_train)))
                            summary.value.add(tag=self.env_id+'on_cur_train/average_@average_reward_per_step@',
                                              simple_value=float(np.mean(self.sum_reward_dic_on_cur_train)))
                            '''summary mo'''
                            summary.value.add(tag=self.env_id+'on_cur_train/average_@sum_mo_per_step@',
                                              simple_value=float(np.mean(self.sum_mo_dic_on_cur_train)))
                            summary.value.add(tag=self.env_id+'on_cur_train/average_@average_mo_per_step@',
                                              simple_value=float(np.mean(self.average_mo_dic_on_cur_train)))
                            self.summary_writer.add_summary(summary, self.cur_training_step)
                            self.summary_writer.flush()

                            '''reset reward record'''
                            self.sum_reward_dic_on_cur_train = []
                            self.average_reward_dic_on_cur_train = []

                            '''reset mo record'''
                            self.sum_mo_dic_on_cur_train = []
                            self.average_mo_dic_on_cur_train = []

                            '''tell outside: we are going to predict on the next run'''
                            self.predicting = True

                            '''update'''
                            self.cur_training_step += 1
                            self.cur_predicting_step += 1

                            if self.cur_predicting_step >= (self.step_total-2):

                                '''on line terminating'''

                                '''record the mo_mean for each subject'''
                                self.save_mo_result()

                                print('on line run meet end, terminate and write done signal')
                                self.terminate_this_worker()

                        else:

                            '''is reward has not been trained to a acceptable range'''

                            '''record reward in this episode run before reset to start point'''
                            self.average_reward_dic_on_cur_train += [np.mean(self.reward_dic_on_cur_episode)]
                            self.sum_reward_dic_on_cur_train += [np.sum(self.reward_dic_on_cur_episode)]

                            '''record mo in this episode run before reset to start point'''
                            self.average_mo_dic_on_cur_train += [np.mean(self.mo_dic_on_cur_episode)]
                            self.sum_mo_dic_on_cur_train += [np.sum(self.mo_dic_on_cur_episode)]

                            '''tell out side: we are not going to predict'''
                            self.predicting = False

                        '''reset anyway since cur_step beyond cur_training_step'''
                        self.reset()
                        done = True

                elif self.predicting is True:

                    '''if is predicting'''

                    if (self.cur_step > self.cur_predicting_step) or (self.if_run_baseline is True):

                        '''if cur_step run beyond cur_predicting_step, means already make a prediction on this step'''

                        '''summary'''
                        summary = tf.Summary()

                        '''summary reward'''
                        summary.value.add(tag=self.env_id+'on_cur_prediction/@sum_reward_per_step@',
                                          simple_value=float(np.sum(self.reward_dic_on_cur_episode)))
                        summary.value.add(tag=self.env_id+'on_cur_prediction/@average_reward_per_step@',
                                          simple_value=float(np.mean(self.reward_dic_on_cur_episode)))
                        summary.value.add(tag=self.env_id+'on_cur_prediction/@reward_for_predicting_step@',
                                          simple_value=float(self.reward_dic_on_cur_episode[-1]))

                        '''summary mo'''
                        summary.value.add(tag=self.env_id+'on_cur_prediction/@sum_mo_per_step@',
                                          simple_value=float(np.sum(self.mo_dic_on_cur_episode)))
                        summary.value.add(tag=self.env_id+'on_cur_prediction/@average_mo_per_step@',
                                          simple_value=float(np.mean(self.mo_dic_on_cur_episode)))
                        mo_on_cur_prediction = self.mo_dic_on_cur_episode[-1]
                        self.mo_on_prediction_dic += [mo_on_cur_prediction]
                        summary.value.add(tag=self.env_id+'on_cur_prediction/@mo_for_predicting_step@',
                                          simple_value=float(mo_on_cur_prediction))
                        summary.value.add(tag=self.env_id+'on_cur_prediction/@average_mo_till_predicting_step@',
                                          simple_value=float(np.mean(self.mo_on_prediction_dic)))

                        self.summary_writer.add_summary(summary, self.cur_predicting_step)
                        self.summary_writer.flush()

                        if self.if_run_baseline is True:

                            if self.cur_step > self.cur_predicting_step:

                                '''if we are running baseline and over the cur_predicting_step, terminate here'''

                                '''record the mo_mean for each subject'''
                                self.save_mo_result()

                                print('on line run meet end, terminate and write done signal')
                                self.terminate_this_worker()

                        else:

                            '''we are not running baseline'''

                            '''tell out side: we are not going to predict'''
                            self.predicting = False

                            '''reset'''
                            self.reset()
                            done = True


        if self.mode is 'off_line':
            return self.cur_observation, reward, done, self.cur_cc, self.max_cc, v_lable
        elif self.mode is 'on_line':
            return self.cur_observation, reward, done, self.cur_cc, self.max_cc, v_lable, self.predicting

    def terminate_this_worker(self):

        '''send signal to terminating this worker'''
        from config import worker_done_signal_dir, worker_done_signal_file

        while True:
            try:
                done_sinal_dic = np.load(worker_done_signal_dir+worker_done_signal_file)['done_sinal_dic']
                break
            except Exception, e:
                print(str(Exception)+": "+str(e))
                time.sleep(1)

        done_sinal_dic=np.append(done_sinal_dic, [[self.env_id_num,self.subject]], axis=0)

        while True:
            try:
                np.savez(worker_done_signal_dir+worker_done_signal_file,
                         done_sinal_dic=done_sinal_dic)
                break
            except Exception, e:
                print(str(Exception)+": "+str(e))
                time.sleep(1)

        while True:
            print('this worker is waiting to be killed')
            time.sleep(1000)

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
            if self.mode is 'off_line':
                self.agent_result_saver += [copy.deepcopy(fixation2salmap(fixation=[[self.cur_lon,self.cur_lon]],
                                                                          mapwidth=self.heatmap_width,
                                                                          mapheight=self.heatmap_height))]
            elif self.mode is 'on_line':
                print('not implement')
                import sys
                sys.exit(0)
    def load_heatmaps(self, name):

        heatmaps = []
        for step in range(self.step_total):

            try:
                file_name = '../../'+self.data_base+'/'+name+'/'+self.env_id+'_'+str(step)+'.jpg'
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

    def save_mo_result(self):

        '''
            Description: save mo result to result dir
        '''

        mo_mean = np.mean(self.mo_on_prediction_dic)
        from config import final_log_dir,if_run_baseline
        if if_run_baseline:
            from config import baseline_type
            self.record_mo_file_name = baseline_type
        else :
            self.record_mo_file_name = "on_line_model"
        with open(final_log_dir+self.record_mo_file_name+"_mo_mean.txt","a") as f:
            f.write("%s\tsubject[%s]:\t%s\n"%(self.env_id,self.subject,mo_mean))
