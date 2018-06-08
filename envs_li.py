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
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq,kmeans,whiten
import subprocess
import urllib
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from vrplayer import get_view
import move_view_lib
import suppor_lib
import tensorflow as tf
import imageio
import config
import cc
import MeanOverlap

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

        self.task = task
        self.summary_writer = summary_writer
        self.env_id = env_id
        from config import game_dic
        self.env_id_num = game_dic.index(self.env_id)

        from config import reward_estimator
        self.reward_estimator = reward_estimator

        from config import mode
        config.mode = mode

        self.subject = subject

        '''load config'''
        self.config()

        '''reset'''
        self.observation = self.reset()

    def get_observation(self):

        '''interface to get view'''
        self.cur_observation = get_view(
            input_width=self.video_size_width,
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
            file_=config.database_path+'/' + self.env_id + '.yuv'
        )

    def config(self):

        '''observation_space'''
        from config import observation_space
        self.observation_space = observation_space

        '''set all temp dir for this worker'''
        if (config.mode is 'off_line') or (config.mode is 'data_processor'):
            self.temp_dir = config.log_dir+"/temp/get_view/w_" + str(self.task) + '/' + str(self.env_id)
        elif config.mode is 'on_line':
            self.temp_dir = config.log_dir+"/temp/get_view/g_" + str(self.env_id) + '_s_' + str(self.subject)
        '''clear temp dir for this worker'''
        subprocess.call(["rm", "-r", self.temp_dir])
        subprocess.call(["mkdir", "-p", self.temp_dir])

        '''some bug in the frame read for some videos'''
        if(self.env_id=='Dubai'):
            self.frame_bug_offset = 540
        else:
            self.frame_bug_offset = 10

        self.subjects_total, self.data_total, self.subjects, _ = suppor_lib.get_subjects(
            data = sio.loadmat(
                config.database_path+'/FULLdata_per_video_frame.mat'
            )[self.env_id],
        )

        self.reward_dic_on_cur_episode = []

        if config.mode is 'on_line':
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
        filename = config.database_path+'/' + self.env_id + '.mp4'
        video = imageio.get_reader(filename,  'ffmpeg')

        self.frame_per_second = float(video._meta['fps'])
        self.frame_total = float(video._meta['nframes'])
        self.frame_size = (video._meta['source_size'])
        self.video_size_width = int(self.frame_size[0])
        self.video_size_heigth = int(self.frame_size[1])

        self.data_total = float(self.data_total)
        self.second_total = self.frame_total / self.frame_per_second
        self.data_per_frame = self.data_total / self.frame_total

        '''compute step lenth from data_tensity'''
        from config import data_tensity
        self.second_per_step = max(data_tensity/self.frame_per_second, data_tensity/self.data_per_frame/self.frame_per_second)
        self.frame_per_step = self.frame_per_second * self.second_per_step
        self.data_per_step = self.data_per_frame * self.frame_per_step

        '''compute step_total'''
        self.step_total = int(self.data_total / self.data_per_step)

        '''set fov range'''
        from config import view_range_lon, view_range_lat
        self.view_range_lon = view_range_lon
        self.view_range_lat = view_range_lat

        self.episode = 0

        self.max_cc_cur_video = 0.0
        self.cc_cur_video = 0.0

        '''salmap'''
        self.heatmap_height = 180
        self.heatmap_width = 360

        if config.mode is 'data_processor':
            self.data_processor()

        '''load ground-truth heat map'''
        load_dir  = '{}/groundtruth_heatmaps/{}'.format(
            config.log_dir,
            self.env_id,
        )
        self.gt_heatmaps = self.load_heatmaps(load_dir)

        self.log_thread_config()

    def data_processor(self):
        print('data process start: '+config.data_processor_id)

        if config.data_processor_id in ['mp4_to_yuv']:
            mp4_filename = config.database_path+'/' + self.env_id + '.mp4'
            yuv_filename = config.database_path+'/' + self.env_id + '.yuv'
            subprocess.call(['ffmpeg', '-i', mp4_filename, yuv_filename])

        if config.data_processor_id in ['generate_groundtruth_heatmaps']:
            save_dir  = '{}/groundtruth_heatmaps/{}'.format(
                config.log_dir,
                self.env_id,
            )
            subprocess.call(["mkdir", "-p", save_dir])

            heatmaps_cur_video = []

            for step_i in range(self.step_total):
                data_i = int(round((step_i)*self.data_per_step))

                '''generate groundtruth heatmaps'''
                HM_positions_for_all_subjects_at_cur_step = []
                for subject_i in range(self.subjects_total):
                    HM_positions_for_all_subjects_at_cur_step += [self.subjects[subject_i].data_frame[data_i].p]
                HM_positions_for_all_subjects_at_cur_step = np.stack(HM_positions_for_all_subjects_at_cur_step)
                temp = suppor_lib.fixation2salmap(
                    fixation=HM_positions_for_all_subjects_at_cur_step,
                    mapwidth=self.heatmap_width,
                    mapheight=self.heatmap_height,
                )
                heatmaps_cur_video += [temp]
                print('heatmaps_cur_video: {}'.format(np.shape(heatmaps_cur_video)))
            heatmaps_cur_video = np.stack(heatmaps_cur_video)

            '''save predicted heatmaps as image'''
            self.save_heatmaps(
                save_dir = save_dir,
                heatmaps = heatmaps_cur_video,
            )

        if config.data_processor_id in ['generate_groundtruth_scanpaths']:
            save_dir  = '{}/groundtruth_scanpaths/{}'.format(
                config.log_dir,
                self.env_id,
            )
            subprocess.call(["mkdir", "-p", save_dir])

            scanpaths_cur_video = []

            for step_i in range(self.step_total):
                data_i = int(round((step_i)*self.data_per_step))

                '''generate groundtruth scanpaths'''
                HM_positions_for_all_subjects_at_cur_step = []
                for subject_i in range(self.subjects_total):
                    HM_positions_for_all_subjects_at_cur_step += [self.subjects[subject_i].data_frame[data_i].p]
                HM_positions_for_all_subjects_at_cur_step = np.stack(HM_positions_for_all_subjects_at_cur_step)
                scanpaths_cur_video += [HM_positions_for_all_subjects_at_cur_step]
                print('scanpaths_cur_video: {}'.format(np.stack(scanpaths_cur_video).shape))
            scanpaths_cur_video = np.stack(scanpaths_cur_video)

            '''save groundtruth scanpaths as npy, [subjects, steo, 2]'''
            np.save(
                '{}/all.npy'.format(
                    save_dir
                ),
                scanpaths_cur_video,
            )

        # if data_processor_id is 'minglang_mp4_to_yuv':
        #     print('sssss')
        #     from config import game_dic_new_all
        #     for i in range(len(game_dic_new_all)):
        #         # print(game_dic_new_all[i])
        #         if i >= 0 and i <= 0: #len(game_dic_new_all)
        #             # file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+str(game_dic_new_all[i])+'.mp4'
        #             file_out_1 = '/media/minglang/YuhangSong_1/ff/vr_yuv/'+"Let'sNotBeAloneTonight"+'.yuv'
        #             file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+"Let'sNotBeAloneTonight"+'.mp4'
        #             self.video = cv2.VideoCapture(file_in_1)
        #             input_width_1 = int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        #             input_height_1 = int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        #             self.mp4_to_yuv(input_width_1,input_height_1,file_in_1,file_out_1)
        #             print('end processing: ',file_out_1)
        #
        #     # print('len_game_dic_new_all: ',len(game_dic_new_all))
        #     # print('get_view')
        #
        #     # print(game_dic_new_all)
        #
        # if data_processor_id is 'minglang_mp4_to_jpg':
        #     from config import game_dic_new_all
        #     for i in range(len(game_dic_new_all)):
        #         # print(game_dic_new_all[i])
        #         if i >= 1 and i <= len(game_dic_new_all): #len(game_dic_new_all)
        #             # file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+str(game_dic_new_all[i])+'.mp4'
        #             # file_out_1 = '/media/minglang/YuhangSong_1/ff/vr_yuv/'+"Let'sNotBeAloneTonight"+'.yuv'
        #             # file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+"Let'sNotBeAloneTonight"+'.mp4'
        #             file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+str(game_dic_new_all[i])+'.mp4'
        #             file_out_1 = '/media/minglang/YuhangSong_1/ff/vr_bms_jpg/'+str(game_dic_new_all[i])+'.yuv'
        #
        #             video = cv2.VideoCapture(file_in_1)
        #             self.video = video
        #             self.frame_per_second = round(video.get(cv2.cv.CV_CAP_PROP_FPS))
        #             self.frame_total = round(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        #
        #             for frame_i in range(int(self.frame_total)):
        #
        #                 try:
        #                     rval, frame = self.video.read()
        #                     # here minglang 1
        #                     cv2.imwrite('/media/minglang/YuhangSong_1/ff/vr_bms_jpg/'+str(game_dic_new_all[i])+'_'+str(frame_i)+'.jpg',frame)
        #                     print(frame_i)
        #                 except Exception, e:
        #                     print('failed on this frame, continue')
        #                     print Exception,":",e
        #                     continue
        #
        #             print('end processing: ',file_in_1,self.frame_per_second,self.frame_total)
        #
        # if data_processor_id is 'minglang_obdl_cfg':
        #     from config import game_dic_new_all
        #     for i in range(len(game_dic_new_all)):
        #         # print(game_dic_new_all[i])
        #
        #         if i >= 100 and i <=  len(game_dic_new_all): #len(game_dic_new_all)
        #             # file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+str(game_dic_new_all[i])+'.mp4'
        #             # file_out_1 = '/media/minglang/YuhangSong_1/ff/vr_yuv/'+"Let'sNotBeAloneTonight"+'.yuv'
        #             # file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+"Let'sNotBeAloneTonight"+'.mp4'
        #             file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+str(game_dic_new_all[i])+'.mp4'
        #             CONFIG_FILE = '/media/minglang/YuhangSong_1/ff/obdl_vr_new/'+str(game_dic_new_all[i])+'.cfg'
        #
        #             # # get the paramters
        #             video = cv2.VideoCapture(file_in_1)
        #             self.video = video
        #             self.frame_per_second = int(round(video.get(cv2.cv.CV_CAP_PROP_FPS)))
        #             self.frame_total = int(round(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)))
        #             NAME = game_dic_new_all[i]
        #             FRAMESCOUNT = self.frame_total
        #             FRAMERATE = self.frame_per_second
        #             IMAGEWIDTH = int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        #             IMAGEHEIGHT = int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        #
        #             # write the paramters throuh cfg
        #             # conf = ConfigParser.ConfigParser()
        #             # cfgfile = open(CONFIG_FILE,'w')
        #             # # conf.add_section("")
        #
        #             # write through txt
        #             f_config = open(CONFIG_FILE,"w")
        #             f_config.write("NAME\n")
        #             f_config.write(str(game_dic_new_all[i])+'\n')
        #             f_config.write("FRAMESCOUNT\n")
        #             f_config.write(str(FRAMESCOUNT)+'\n')
        #             f_config.write("FRAMERATE\n")
        #             f_config.write(str(FRAMERATE)+'\n')
        #             f_config.write("IMAGEWIDTH\n")
        #             f_config.write(str(IMAGEWIDTH)+'\n')
        #             f_config.write("IMAGEHEIGHT\n")
        #             f_config.write(str(IMAGEHEIGHT)+'\n')
        #             f_config.close()
        #
        #         #one video and one cfg in one file
        #         if i >= 0 and i <= len(game_dic_new_all): #len(game_dic_new_all)
        #             cfg_file = '/media/minglang/YuhangSong_1/ff/obdl_vr_new/obdl_vr_new/'+str(game_dic_new_all[i])
        #             os.makedirs(cfg_file)
        #             file_in_1 = '/media/minglang/YuhangSong_1/ff/vr_new/'+str(game_dic_new_all[i])+'.mp4'
        #             shutil.copy(file_in_1,cfg_file)
        #             CONFIG_FILE = '/media/minglang/YuhangSong_1/ff/obdl_vr_new/'+str(game_dic_new_all[i])+'.cfg'
        #             shutil.copy(CONFIG_FILE,cfg_file)
        #             print("os.makedirs(cfg_file)")


        raise Exception('data process end, programe terminate.')

    def log_thread_config(self):

        self.if_log_scan_path_real_time = config.if_log_scan_path_real_time
        self.if_log_results = config.if_log_results

        if self.if_log_results:

            if config.mode is 'off_line':
                '''cc record'''
                self.agent_heatmap_saver_cur_episode = []
                self.agent_heatmap_saver_multiple_episodes = []
                '''scanpath location record'''
                self.agent_scanpath_saver_cur_episode = []
                self.agent_scanpath_saver_multiple_episodes = []

                if config.mode is 'off_line':
                    self.predicted_fixtions_num = config.predicted_fixation_num
                    self.if_log_results_interval = config.log_results_interval

    def reset(self):

        '''reset cur_step and cur_data'''
        self.cur_step = 0
        self.cur_data = 0

        self.reward_dic_on_cur_episode = []

        if config.mode is 'on_line':
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
        if config.mode is 'off_line':
            subject_code = np.random.choice(a=subject_dic_code)
        elif config.mode is 'on_line':
            subject_code = 0
        self.cur_lon = self.subjects[subject_code].data_frame[0].p[0]
        self.cur_lat = self.subjects[subject_code].data_frame[0].p[1]

        '''set observation_now to the first frame'''
        self.get_observation()

        self.last_observation = None

        self.log_thread_reset()

        return self.cur_observation

    def log_thread_reset(self):

        if self.if_log_scan_path_real_time:
            plt.figure(str(self.env_id)+'_scan_path')
            plt.clf()

        if self.if_log_results:

            if np.array(self.agent_heatmap_saver_cur_episode).shape[0] == 0:
                return

            if config.mode is 'off_line':

                self.agent_heatmap_saver_multiple_episodes += [np.array(self.agent_heatmap_saver_cur_episode)]
                self.agent_scanpath_saver_multiple_episodes += [np.array(self.agent_scanpath_saver_cur_episode)]

                self.agent_heatmap_saver_cur_episode = []
                self.agent_scanpath_saver_cur_episode = []

                if len(self.agent_heatmap_saver_multiple_episodes) > self.predicted_fixtions_num:

                    '''if stack full, pop out the oldest data'''
                    self.agent_heatmap_saver_multiple_episodes.pop(0)
                    self.agent_scanpath_saver_multiple_episodes.pop(0)

                    if self.episode%self.if_log_results_interval is 0:

                        print('computing CC')

                        ccs_on_step_i = []
                        heatmaps_cur_video = []
                        all_scanpath_locations = []

                        for step_i in range(self.step_total):

                            '''generate predicted salmap'''
                            temp = np.stack(self.agent_heatmap_saver_multiple_episodes)[:,step_i]
                            temp = np.sum(temp,axis=0)
                            temp = temp / np.max(temp)
                            heatmaps_cur_video += [temp]

                            'save the scanpath locations'
                            sc_locations_one_step = np.stack(self.agent_scanpath_saver_multiple_episodes)[:,step_i]
                            all_scanpath_locations += [sc_locations_one_step]

                            ccs_on_step_i += [(cc.calc_score(
                                gtsAnn = self.gt_heatmaps[step_i],
                                resAnn = heatmaps_cur_video[step_i],
                            ))]
                            print('cc on step {} is {}'.format(
                                step_i,
                                ccs_on_step_i[step_i],
                            ))

                        self.cc_cur_video = np.mean(np.stack(ccs_on_step_i))
                        print('cc_cur_video is '+str(self.cc_cur_video))
                        if self.cc_cur_video > self.max_cc_cur_video:
                            print('new max cc found: '+str(self.cc_cur_video)+', recording cc and heatmaps')
                            self.max_cc_cur_video = self.cc_cur_video
                            self.heatmaps_of_max_cc_cur_video = np.stack(heatmaps_cur_video)
                            self.scanpath_of_max_cc_cur_video = np.stack(all_scanpath_locations)

                            save_heatmap_dir  = '{}/predicted_heatmaps/{}'.format(
                                config.log_dir,
                                self.env_id,
                            )
                            save_scanpath_dir = '{}/predicted_scanpath/{}'.format(
                                config.log_dir,
                                self.env_id,
                            )
                            subprocess.call(["mkdir", "-p", save_heatmap_dir])
                            subprocess.call(["mkdir", "-p", save_scanpath_dir])

                            '''save predicted heatmaps as image'''
                            self.save_heatmaps(
                                save_dir = save_heatmap_dir,
                                heatmaps = self.heatmaps_of_max_cc_cur_video,
                            )

                            '''save predicted scanpath
                            shape: [agent, step, 2]'''
                            np.save(
                                '{}/all.npy'.format(
                                    save_scanpath_dir
                                ),
                                self.scanpath_of_max_cc_cur_video.shape,
                            )

    def step(self, action, v):

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
        if (update_frame_success==False) or (update_data_success==False):

            '''terminating'''
            self.reset()
            reward = 0.0
            mo = 0.0
            done = True
            v_lable = 0.0

        else:

            if config.mode is 'on_line':

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
            last_prob, distance_per_data = suppor_lib.get_prob(
                lon=self.last_lon,
                lat=self.last_lat,
                theta=action * 45.0,
                subjects=self.subjects,
                subjects_total=self.subjects_total,
                cur_data=self.last_data,
            )
            '''rescale'''
            distance_per_step = distance_per_data * self.data_per_step
            '''convert v to degree'''
            degree_per_step = distance_per_step / math.pi * 180.0
            '''set v_lable'''
            v_lable = degree_per_step

            '''move view, update cur_lon and cur_lat, the standard procedure of rl'''
            v_used_to_step = v

            self.cur_lon, self.cur_lat = move_view_lib.move_view(
                cur_lon=self.last_lon,
                cur_lat=self.last_lat,
                direction=action,
                degree_per_step=v_used_to_step,
            )
            self.last_action = action

            '''produce reward'''
            if self.reward_estimator is 'trustworthy_transfer':
                reward = last_prob
            elif self.reward_estimator is 'cc':
                cur_heatmap = suppor_lib.fixation2salmap(
                    fixation=np.array([[self.cur_lon, self.cur_lat]]),
                    mapwidth=self.heatmap_width,
                    mapheight=self.heatmap_height,
                )
                reward = cc.calc_score(self.gt_heatmaps[self.cur_step], cur_heatmap)

            if config.mode is 'on_line':

                '''compute MO'''
                mo_calculator = MeanOverlap.MeanOverlap(self.video_size_width,
                                            self.video_size_heigth,
                                            65.5/2,
                                            3.0/4.0)
                mo = mo_calculator.calc_mo_deg((self.cur_lon,self.cur_lat),(self.subjects[0].data_frame[self.cur_data].p[0],self.subjects[0].data_frame[self.cur_data].p[1]),is_centered = True)
                self.mo_dic_on_cur_episode += [mo]

            '''smooth reward, if we have last_action'''
            if self.last_action is not None:

                '''compute smooth reward'''
                action_difference = abs(action-self.last_action)
                from config import direction_num
                if action_difference > (direction_num/2):
                    action_difference -= (direction_num/2)
                from config import reward_smooth_discount_to
                reward *= (1.0-(action_difference*(1.0-reward_smooth_discount_to)/(direction_num/2)))

            '''record'''
            self.reward_dic_on_cur_episode += [reward]

            '''All reward and scores has been computed, we now consider if we want to drawback the position'''
            if config.mode in ['on_line']:

                if self.predicting:

                    '''if we are predicting we are actually feeding the model so that we can produce
                    a prediction with the experiences already experienced by the human.'''
                    self.cur_lon = self.subjects[0].data_frame[self.cur_data].p[0]
                    self.cur_lat = self.subjects[0].data_frame[self.cur_data].p[1]

            '''after pull the position, get observation
            update observation_now'''
            self.get_observation()

            '''normally, we donot judge done when we in this'''
            done = False

            '''core part for online'''
            if config.mode in ['on_line']:

                if not self.predicting:

                    if self.cur_step > self.cur_training_step:

                        '''if step is out of training range'''
                        if (np.mean(self.reward_dic_on_cur_episode) > self.train_to_reward) or (np.mean(self.mo_dic_on_cur_episode) > self.train_to_mo) or (len(self.sum_reward_dic_on_cur_train)>self.train_to_episode):
                            '''if reward is trained to a acceptable range or trained episode exceed a range'''

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

                            if self.cur_predicting_step >= (self.step_total-1):
                                '''on line terminating'''
                                '''record the mo_mean for each subject'''
                                self.save_mo_result()

                                print('on line run meet end, terminate and write done signal')
                                self.terminate_this_worker()

                        else:
                            '''if has not been trained to a acceptable range'''

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

        if config.mode is 'off_line':
            return self.cur_observation, reward, done, self.cc_cur_video, self.max_cc_cur_video, v_lable
        elif config.mode is 'on_line':
            return self.cur_observation, reward, done, self.cc_cur_video, self.max_cc_cur_video, v_lable, self.predicting

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

        if self.if_log_scan_path_real_time:
            plt.figure(str(self.env_id)+'_scan_path')
            plt.scatter(self.cur_lon, self.cur_lat, c='r')
            plt.scatter(-180, -90)
            plt.scatter(-180, 90)
            plt.scatter(180, -90)
            plt.scatter(180, 90)
            plt.pause(0.00001)

        if self.if_log_results:
            if config.mode in ['off_line']:
                self.agent_heatmap_saver_cur_episode += [suppor_lib.fixation2salmap(
                    fixation  = np.array([[self.cur_lon,self.cur_lat]]),
                    mapwidth  = self.heatmap_width,
                    mapheight = self.heatmap_height,
                )]
                self.agent_scanpath_saver_cur_episode +=[np.array(
                    [self.cur_lon,self.cur_lat]
                )]
            elif config.mode is 'on_line':
                raise Exception('Do not set if_log_results=True when using online mode')

    def save_heatmaps(self, save_dir, heatmaps):
        heatmaps = (heatmaps * 255.0).astype(np.uint8)
        for step_i in range(self.step_total):
            imageio.imwrite(
                '{}/{}.jpg'.format(
                    save_dir,
                    step_i,
                ),
                heatmaps[step_i]
            )

    def load_heatmaps(self, load_dir):

        heatmaps = []
        for step_i in range(self.step_total):
            try:
                temp = cv2.imread(
                    '{}/{}.jpg'.format(
                        load_dir,
                        step_i,
                    ),
                    cv2.CV_LOAD_IMAGE_GRAYSCALE,
                )
                temp = cv2.resize(temp,(self.heatmap_width, self.heatmap_height))
                temp = temp / 255.0
                heatmaps += [temp]
            except Exception,e:
                raise Exception(Exception,":",e)

        heatmaps = np.stack(heatmaps)
        print('load heatmaps from '+load_dir+' done, size: '+str(np.shape(heatmaps)))

        return heatmaps

    def save_mo_result(self):

        raise Exception('Dirty code....')
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
