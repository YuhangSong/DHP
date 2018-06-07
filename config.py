'''set this path to the folder containing YUV, MP4 and MAT of our database'''
database_path = '/media/yuhangsong/YuhangSong_1/env/ff/vr_new'

if_restore_model = True
if if_restore_model is True:
    model_to_restore = "model_to_restore/model.ckpt-8496809"

log_dir = "../results/reproduce_4"
'''if clear the logdir before running'''
clear_run = False
import subprocess
if clear_run:
    '''if clear_run, clear the path before create the path'''
    raw_input('You are clearing {}, is that what you want? If not, press Ctrl+C and set clear_run=False in config.py'.format(log_dir))
    subprocess.call(["rm", "-r", log_dir])
subprocess.call(["mkdir", "-p", log_dir])

'''set model structure'''
conv_depth = 4
# consi_depth is from another work of mine:
# https://arxiv.org/abs/1710.10036,
# ignore it (keep it 1), if you are not interested
consi_depth = 1
# if set consi_depth=1, only lstm_size[0] matters
lstm_size = [288,128,32]

'''
Description: how many steps do you want to run before update the model
Note:
    20 is empirically good for single task
    larger value may benifit the multi-task performence
'''
update_step = 40

'''
Description: select mode
Availible: off_line, on_line, data_processor
'''
mode = 'data_processor'

if mode in ['off_line']:
    procedure = 'train'
    # Note that for online settings, there is no separation for training and teseting

import dataset_config
if mode in ['off_line']:
    if procedure in ['train']:
        game_dic = dataset_config.train_set
    elif procedure in ['test']:
        game_dic = dataset_config.test_set
elif mode in ['on_line']:
    game_dic = dataset_config.test_set
elif mode in ['data_processor']:
    game_dic = dataset_config.train_set + dataset_config.test_set

num_games = len(game_dic)

'''
number of direction to action
this is because we use discrete control
'''
direction_num = 8

if_log_scan_path_real_time = False
if_log_results = False

if if_log_results is True:

    '''number of workflow to produce salmap,
    you set it to any value, but we set it to num_subjects,
    refer to Section xx in https://arxiv.org/abs/1710.10755'''
    predicted_fixation_num = dataset_config.num_subjects
    log_results_interval = 5

'''
    Description: config env
'''
data_tensity = 10.0
view_range_lon = 110
view_range_lat = 113
final_discount_to = 10**(-4)
from numpy import zeros
observation_space = zeros((42, 42, 1))
reward_estimator = 'trustworthy_transfer' # availible: trustworthy_transfer, cc
heatmap_sigma = 'sigma_half_fov' # availible: my_sigma, sigma_half_fov
reward_smooth_discount_to = 1.0 # set to 1.0 to disable reward smooth
if_normalize_v_lable = True

if mode is 'off_line':

    '''
        Description: specific settings for off_line mode
    '''

elif mode is 'data_processor':

    '''
        Description: specific settings for data_processor mode
    '''

    '''
        Description: what data_processor you are doing
        Availible:
                    mp4_to_yuv (yuhang)
                    generate_groundtruth_heatmaps (yuhang)
                    generate_groundtruth_scanpaths (yuhang)
                    minglang_mp4_to_yuv (minglang)
                    compute_consi (haochen)
                    minglang_mp4_to_jpg (minglang)
                    minglang_obdl_cfg (minglang)
    '''
    data_processor_id = 'generate_groundtruth_heatmaps'

    if data_processor_id is 'compute_consi':

        if project is 'f' and mode is 'data_processor':

            '''
                Description: compute_consi config
            '''

            fov_degree = 6
            no_moving_gate = 0.0001
            compute_lon_inter = fov_degree / 2
            compute_lat_inter = fov_degree / 2
            frame_gate = 20
            MaxCenterNum = 4
            NumDirectionForCluster = 8
            DirectionInter = 360 / NumDirectionForCluster

elif mode is 'on_line':

    '''
        Description: specific settings for on_line mode
    '''

    '''
        Description: conditions to terminate and move on the on_line train
    '''
    train_to_reward = 0.2 # set to 1.0 to disable it
    train_to_mo = 0.8 # set to 1.0 to disable it
    train_to_episode = 500 # too big would make some train hard to end, for some subjects is too hard to learn

# my_sigma = (11.75+13.78)/2
my_sigma = 7
import math
sigma_half_fov = 7, # 51.0 / (math.sqrt(-2.0*math.log(0.5)))
check_worker_done_time = 5
if mode is 'on_line':
    num_workers_one_run = num_workers_one_run_proper
    if debugging is True:
        num_workers_one_run = 2
    if data_base is 'vr':
        num_subjects = 40
    elif data_base is 'vr_new':
        num_subjects = 58
    worker_done_signal_dir = 'temp/worker_done_signal_dir/'
    worker_done_signal_file = 'worker_done_signal.npz'
