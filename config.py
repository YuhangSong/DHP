'''
    Description: cluster config
'''
cluster_current = 1

'''
    Description: which project you want to run
    Availible: g, f
'''
project = 'g'

'''
    Description: set True if you are debugging
    Note:
        if set True, following things will happen:
        1,the workers ran at one time will be fewer to enable faster running
        3,the directory to store results data will have a temp_run pre-fix, so that
          it is clear that the results is not for analysis
        2,the directory to store results data will be cleaned
'''
debugging = True

if debugging is True:

    '''
        Description: cut game_dic to a smaller range to debug
        Availible:
    '''
    debugging_range = [0,1]

'''
    Description: if restore model
'''
if_restore_model = True

if if_restore_model is True:

    '''
        Description: which mode you want to restore
    '''
    model_to_restore = "model_to_restore/model.ckpt-8496809"

'''
    Description: set your log dir to store results data
'''
basic_log_dir = project+"_101"
log_dir = "fix"


'''
    Description:if separate game_dic
'''
if_separate_game_dic = False

if if_separate_game_dic :
    separate_start_game_index_from = 0  #  set the start game index, set to -1 to be extrame
    separate_start_game_index_to = 5  #  set the start game index, set to -1 to be extrame


'''
    Description: set model structure
'''
conv_depth = 4
consi_depth = 1
lstm_size = [288,128,32]

'''
    Description: how many steps do you want to run before update the model
    Note:
        20 is empirically good for single task
        larger value may benifit the multi-task performence
'''
update_step = 20

'''
    Description: mix experiences from different workers
    Note:
        unavailible
'''
if_mix_exp = False

'''
    Description: auto normalize step reward
    Note:
        unavailible
'''
if_reward_auto_normalize = False


if project is 'f':

    '''
        Description: specific settings for project g
    '''

    from g_game_dic import *
    game_dic = g_game_dic_test_single_pong # specific game dic

    '''
        Description:
    '''
    model = None

elif project is 'f':

    '''
        Description: specific settings for project f
    '''

    '''
        Description: select data_base you are running on
        Availible: vr, vr_new
    '''
    data_base = 'vr_new'

    if data_base is 'vr_new':
        from f_game_dic import f_game_dic_new_all, f_game_dic_new_test
        game_dic = f_game_dic_new_test # specific game dic
    elif data_base is 'vr':
        from f_game_dic import f_game_dic_all
        game_dic = f_game_dic_all # specific game dic

    '''
        Description: select mode
        Availible: off_line, on_line, data_processor
    '''
    mode = 'off_line'

    '''
        Description: if learning v in the model,
                     if not, the agent will use the ground-truth v to act
    '''
    if_learning_v = True

    '''
        Description: number of direction to action
        Note:
            this is because we use discrete control
    '''
    direction_num = 8

    '''
        Description: config env behaviour
    '''
    if_log_scan_path = False

    if mode is 'off_line':
        if_log_cc = True
    else:
        if_log_cc = False

    if if_log_cc is True:

        '''config for log cc'''
        relative_predicted_fixation_num = 1.0
        relative_log_cc_interval = 0.5


        # relative_predicted_fixation_num = 2.0 / 58.0
        # relative_log_cc_interval = 1.0

    '''
        Description: config env
    '''
    data_tensity = 10
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
            Availible: minglang_mp4_to_yuv
                       compute_consi
                       minglang_mp4_to_jpg
                       minglang_obdl_cfg
        '''
        data_processor_id = 'compute_direction'

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
        if data_processor_id is 'compute_direction':

            if project is 'f' and mode is 'data_processor':

                '''
                    Description: compute_consi config
                '''
                speed_gate = 15/180*3.14 # rads per sec
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

        '''
            Description: if you want to run baseline of the on_line prediction
        '''
        if_run_baseline = False

        if if_run_baseline is True:

            '''
                Description: basic settings for running baseline in on_line mode
            '''

            '''
                Description: select baseline type
                Availible: keep, random
                Note:
                    keep: to keep the direction of last action
                    random: to select the action of direction and v randomly
            '''
            baseline_type = 'keep'

            '''
                Description: v used when runing the baseline
                Note: this value should be a averaged value from all data_base
            '''
            v_used_in_baseline = 0.0745080846


'''
    Description: default config generated from above config
'''

cluster_host                   = ['192.168.226.67', '192.168.226.83', '192.168.226.139', '192.168.1.31','192.168.226.197','192.168.226.83'] # main cluster has to be first
cluster_name                   = ['yuhangsong'    , 'Server'        , 'WorkerR'        , 'xuntian2'    ,'haochen'        ,'Worker4'] # main cluster has to be first
cluster_home                   = ['yuhangsong'    , 's'             , 'irc207'         , 'xuntian2'    ,'s'              ,'s'] # main cluster has to be first
num_workers_one_run_max_dic    = [8               , -1              , -1               , 8             ,16               ,-1]
num_workers_one_run_proper_dic = [8               , 32              , 32               , 8             ,8                ,16]
num_workers_one_run_max = num_workers_one_run_max_dic[cluster_current]
num_workers_one_run_proper = num_workers_one_run_proper_dic[cluster_current]

if project is 'g':

    from g_game_dic import g_game_dic_all
    game_dic_all = g_game_dic_all

if project is 'f':

    use_move_view_lib = 'ziyu'

    if data_base is 'vr_new':
        from f_game_dic import f_game_dic_new_all
        game_dic = f_game_dic_new_all # specific game dic
    elif data_base is 'vr':
        from f_game_dic import f_game_dic_all
        game_dic = f_game_dic_all # specific game dic

    if if_separate_game_dic :
        game_dic = game_dic[separate_start_game_index_from:separate_start_game_index_to]

    my_sigma = (11.75+13.78)/2
    import math
    sigma_half_fov = 51.0 / (math.sqrt(-2.0*math.log(0.5)))
    check_worker_done_time = 5
    if mode is 'off_line' or mode is 'data_processor':
        '''off line run all video together, should constrain the game_dic'''
        if num_workers_one_run_max is not -1:
            game_dic = game_dic[0:num_workers_one_run_max]
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

if if_separate_game_dic :
    if separate_start_game_index_from is -1:
        separate_start_game_index_from = 0
    if separate_start_game_index_to is -1:
        separate_start_game_index_to = len(game_dic)
    game_dic = game_dic[separate_start_game_index_from:separate_start_game_index_to]

if debugging is True:
    status = "temp_run"
    game_dic = game_dic[debugging_range[0]:debugging_range[1]]
else:
    status = ""

final_log_dir = "../../result/"+basic_log_dir+status+"/" + log_dir + status+'/'

if status is "temp_run":
    import subprocess
    subprocess.call(["rm", "-r", final_log_dir])

num_games_global = len(game_dic)
