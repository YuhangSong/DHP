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

        from config import reward_estimator
        self.reward_estimator = reward_estimator

        from config import mode
        self.mode = mode

        self.subject = subject

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
                                        file_='../../'+self.data_base+'/' + self.env_id + '.yuv')

    def config(self):

        '''function to load config'''
        print("=================config=================")

        from config import data_base
        self.data_base = data_base

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
            self.cur_training_step = 0
            self.cur_predicting_step = self.cur_training_step + 1
            self.predicting = False
            from config import train_to_reward
            self.train_to_reward = train_to_reward
            self.sum_reward_dic_on_cur_train = []
            self.average_reward_dic_on_cur_train = []



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
            from config import num_workers_global,cluster_current,cluster_main
            if (self.task%num_workers_global==0) and (cluster_current==cluster_main):
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
        if data_processor_id is 'minglang_mp4_to_jpg':
            print('fffff')
        if data_processor_id is 'compute_consi':
            cov_on_video,valid_circle_exp_per_frame=compute_consi(self.subjects,self.data_total,self.subjects_total)
            store_consi(cov_on_video,valid_circle_exp_per_frame)
        print('=============================data process end, programe terminate=============================')

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
            done = True
            if self.if_learning_v:
                v_lable = 0.0

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

            if (self.mode is 'on_line') and (self.predicting is True):
                '''online and predicting, lon and lat is updated as subjects' ground-truth'''
                '''other procedure may not used by the agent, but still implemented to keep the interface unified'''
                print('predicting run')
                self.cur_lon = self.subjects[0].data_frame[self.cur_data].p[0]
                self.cur_lat = self.subjects[0].data_frame[self.cur_data].p[1]
            else:
                '''move view, update cur_lon and cur_lat, the standard procedure of rl'''
                if self.if_learning_v:
                    self.cur_lon, self.cur_lat = self.view_mover.move_view(direction=action * 45.0,degree_per_step=v)
                    v_lable = degree_per_step
                else:
                    self.cur_lon, self.cur_lat = self.view_mover.move_view(direction=action * 45.0,degree_per_step=degree_per_step)

            '''update observation_now'''
            self.get_observation()

            '''produce reward'''
            if self.reward_estimator is 'trustworthy_transfer':
                reward = last_prob
            elif self.reward_estimator is 'cc':
                cur_heatmap = fixation2salmap(fixation=[[self.cur_lon, self.cur_lat]],
                                              mapwidth=self.heatmap_width,
                                              mapheight=self.heatmap_height)
                from cc import calc_score
                reward = calc_score(self.gt_heatmaps[self.cur_step], cur_heatmap)

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
            self.last_action = action
            self.reward_dic_on_cur_episode += [reward]

            '''normally, we donot judge done when we in this'''
            done = False

            if self.mode is 'on_line':

                if self.predicting is False:

                    '''if is training'''
                    if self.cur_step > self.cur_training_step:

                        '''if step is out of training range'''

                        if np.mean(self.reward_dic_on_cur_episode) > self.train_to_reward:

                            '''if reward is trained to a acceptable range'''

                            '''summary'''
                            summary = tf.Summary()
                            summary.value.add(tag=self.env_id+'on_cur_train/number_of_episodes',
                                              simple_value=float(len(self.sum_reward_dic_on_cur_train)))
                            summary.value.add(tag=self.env_id+'on_cur_train/average_@sum_reward_per_step@',
                                              simple_value=float(np.mean(self.sum_reward_dic_on_cur_train)))
                            summary.value.add(tag=self.env_id+'on_cur_train/average_@average_reward_per_step@',
                                              simple_value=float(np.mean(self.sum_reward_dic_on_cur_train)))
                            self.summary_writer.add_summary(summary, self.cur_training_step)
                            self.summary_writer.flush()

                            '''reset'''
                            self.sum_reward_dic_on_cur_train = []
                            self.average_reward_dic_on_cur_train = []

                            '''tell outside: we are going to predict on the next run'''
                            self.predicting = True

                            '''update'''
                            self.cur_training_step += 1
                            self.cur_predicting_step += 1

                            if self.cur_predicting_step >= self.step_total:

                                '''on line terminating'''
                                print('on line run meet end, terminating..')
                                import sys
                                sys.exit(0)

                        else:

                            '''is reward has not been trained to a acceptable range'''

                            '''record this episode run before reset to start point'''
                            self.average_reward_dic_on_cur_train += [np.mean(self.reward_dic_on_cur_episode)]
                            self.sum_reward_dic_on_cur_train += [np.sum(self.reward_dic_on_cur_episode)]

                            '''tell out side: we are not going to predict'''
                            self.predicting = False

                        '''reset anyway since cur_step beyond cur_training_step'''
                        self.reset()
                        done = True

                else:

                    '''if is predicting'''

                    if self.cur_step > self.cur_predicting_step:

                        '''if cur_step run beyond cur_predicting_step, means already make a prediction on this step'''

                        '''summary'''
                        summary = tf.Summary()
                        summary.value.add(tag=self.env_id+'on_cur_prediction/@sum_reward_per_step@',
                                          simple_value=float(np.sum(self.reward_dic_on_cur_episode)))
                        summary.value.add(tag=self.env_id+'on_cur_prediction/@average_reward_per_step@',
                                          simple_value=float(np.mean(self.reward_dic_on_cur_episode)))
                        summary.value.add(tag=self.env_id+'on_cur_prediction/@reward_for_predicting_step@',
                                          simple_value=float(self.reward_dic_on_cur_episode[-1]))
                        self.summary_writer.add_summary(summary, self.cur_predicting_step)
                        self.summary_writer.flush()

                        '''tell out side: we are not going to predict'''
                        self.predicting = False

                        '''reset'''
                        self.reset()
                        done = True

        if self.mode is 'off_line':
            return self.cur_observation, reward, done, self.cur_cc, self.max_cc, v_lable
        elif self.mode is 'on_line':
            return self.cur_observation, reward, done, self.cur_cc, self.max_cc, v_lable, self.predicting

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
def compute_consi(subjects,num_data_frame,num_subject):
    print('compute_consi')
    '''config'''
    from config import NumDirectionForCluster,frame_gate,fov_degree
    from config import compute_lat_inter,compute_lon_inter
    from config import DirectionInter
    ''''''
    sum_on_video = np.zeros((NumDirectionForCluster))
    count_on_video = 0
    valid_circle_count = 0
    ''''''
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
                            print('!!!!')
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
    valid_circle_exp_per_frame = valid_circle_count * 1.0 / count_on_video
    print('compute_over')
    return cov_on_video,valid_circle_exp_per_frame
def store_consi(cov_on_video,valid_circle_exp_per_frame):
    print("store_consi_result")
    '''config'''
    from config import fov_degree,no_moving_gate,compute_lon_inter,compute_lat_inter
    from config import frame_gate,MaxCenterNum,NumDirectionForCluster
    print_string='\n'
    print_string += '\tfov_degree\t'+str(fov_degree)+'\tno_moving_gate\t'+str(no_moving_gate)+'\tcompute_lon_inter\t'+str(compute_lon_inter)+'\tcompute_lat_inter\t'+str(compute_lat_inter)
    print_string += '\tframe_gate\t'+str(frame_gate)+'\tMaxCenterNum\t'+str(MaxCenterNum)+'\tvalid_circle_exp_per_frame\t'+str(valid_circle_exp_per_frame)
    display_string = ''
    for print_i in range(NumDirectionForCluster):
        print_string += ('\t' + str(cov_on_video[print_i]))
        display_string += ('\t' + str(cov_on_video[print_i]))
    print(display_string)
    f = open('consistence_result.txt','a')
    f.write(print_string)
    f.close()
    print('store_over')
