project = 'f' #availible: g, f

if project is 'g':
    model = None
elif project is 'f':
    data_base = 'vr_new' #availible: vr, vr_new
    mode = 'on_line' #availible: off_line, on_line, data_processor
    if_learning_v = True
    debugging = False
    debugging_range = [0,1]
    if mode is 'data_processor':
        data_processor_id = 'minglang_obdl_cfg'#availible:minglang_mp4_to_yuv,compute_consi,minglang_mp4_to_jpg
                                                 # minglang_obdl_cfg
'''log config'''
if debugging is True:
    '''default setting'''
    status = "temp_run"
else:
    status = ""

basic_log_dir = project+"_17"
log_dir = "1_testing_on_line_auto_conti_restrain_episode_help_all"
final_log_dir = "../../result/"+basic_log_dir+status+"/" + log_dir + status+'/'

if status is "temp_run":
    import subprocess
    subprocess.call(["rm", "-r", final_log_dir])

# '''restore model config'''
# if_restore_model = False
# if if_restore_model is True:
#     model_to_restore = "../../result/model_to_restore/model.ckpt-8496809"

'''cluster config'''
cluster_current = 3

if (mode is 'off_line') or (mode is 'data_processor'):
    cluster_main = 0
    '''worker config'''
    if mode is 'off_line':
        if debugging is True:
            '''default settings'''
            num_workers_local = 3
        else:
            '''default settings'''
            if cluster_current is 0:
                num_workers_local = 16
            elif cluster_current is 1:
                num_workers_local = 2
            elif cluster_current is 2:
                num_workers_local = 2
    elif mode is 'data_processor':
        '''default settings'''
        num_workers_local = 1
elif mode is 'on_line':
    if debugging:
        '''default settings'''
        num_workers_one_run = 3
    else:
        '''default settings'''
        if cluster_current is 0:
            num_workers_one_run = 4
        elif cluster_current is 1:
            num_workers_one_run = 32
        elif cluster_current is 2:
            num_workers_one_run = 32
        elif cluster_current is 3:
            num_workers_one_run = 4


'''model structure'''
if project is 'g':
    conv_depth = 4
    consi_depth = 3
elif project is 'f':
    conv_depth = 4
    consi_depth = 1
lstm_size = [288,128,32] # consi first is to large

'''behaviour config'''
update_step = 20
# if_mix_exp = False
# if_reward_auto_normalize = False
if project is 'f':
    direction_num = 8

if project is 'f':
    '''for env config'''
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
    '''for env behaivour'''
    if mode is 'off_line':
        if_log_scan_path = False
        if_log_cc = True
    elif mode is 'on_line':
        if_log_scan_path = False
        if_log_cc = False

        '''some conditions to terminate and move on the on_line train'''
        train_to_reward = 0.2 # set to 1.0 to disable it
        train_to_mo = 0.9 # set to 1.0 to disable it
        train_to_episode = 500 # too big would make some train hard to end, for some subjects is too hard to learn

    relative_predicted_fixation_num = 1.0
    relative_log_cc_interval = 3.0/40.0

def get_env_dic(env_seq_id):
    import copy
    env_seq_id = copy.deepcopy(env_seq_id)
    print("Total Games:" + str(len(env_seq_id)))
    for i in range(len(env_seq_id)):
        name = ''.join([g.capitalize() for g in env_seq_id[i].split('_')])
        env_seq_id[i] = '{}Deterministic-v3'.format(name)
    return env_seq_id
def get_env_ac_space(env_id):
    from envs import create_atari_env
    return create_atari_env(env_id).action_space.n
def get_env_dic_ac_space(env_id_dic):
    env_dic_ac_space = {}
    for env_id in env_id_dic:
        env_dic_ac_space[env_id] = get_env_ac_space(env_id)
    return env_dic_ac_space

if project is 'g':
    '''game dic'''
    game_dic_test_single_pong = get_env_dic([
        'pong',
    ])
    game_dic_test_multi_pong = get_env_dic([
        'pong', 'breakout',
    ])
    game_dic_shooting = get_env_dic([
        'assault', 'asteroids', 'beam_rider', 'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'atlantis', 'gravitar', 'phoenix', 'pooyan', 'riverraid', 'seaquest', 'space_invaders', 'star_gunner', 'time_pilot', 'zaxxon', 'yars_revenge',
    ])
elif project is 'f':
    game_dic_all=[
        'A380',
        'AcerPredator',
    ]
    game_dic_new_all=[
        'A380',
        'AcerEngine',
        'AcerPredator',
        'AirShow',
        'BFG',
        'Bicycle',
        'BlueWorld',
        'BTSRun',
        'Camping',
        'CandyCarnival',
        'Castle',
        'Catwalks',
        'CMLauncher',
        'CMLauncher2',
        'CS',
        'DanceInTurn',
        'Dancing',
        'DrivingInAlps',
        'Egypt',
        'F5Fighter',
        'Flight',
        'GalaxyOnFire',
        'Graffiti',
        'GTA',
        'Guitar',
        'HondaF1',
        'InsideCar',
        'IRobot',
        'KasabianLive',
        'KingKong',
        'Lion',
        'LoopUniverse',
        'Manhattan',
        'MC',
        'MercedesBenz',
        'Motorbike',
        'Murder',
        'Let\'sNotBeAloneTonight',
        'Orion',
        'Parachuting',
        'Parasailing',
        'Pearl',
        'Predator',
        'ProjectSoul',
        'Rally',
        'RingMan',
        'RioOlympics',
        'Roma',
        'Shark',
        'Skiing',
        'Snowfield',
        'SnowRopeway',
        'SpaceWar',
        'SpaceWar2',
        'Square',
        'StarryPolar',
        'StarWars',
        'StarWars2',
        'Stratosphere',
        'StreetFighter',
        'Sunset',
        'Supercar',
        'SuperMario64',
        'Surfing',
        'SurfingArctic',
        'Symphony',
        'TalkingInCar',
        'Terminator',
        'TheInvisible',
        'Village',
        'VRBasketball',
        'WaitingForLove',
        'Waterfall',
        'Waterskiing',
        'WesternSichuan',
        'Yacht',
    ]

'''env config'''
if project is 'g':
    game_dic = game_dic_test_multi_pong # specific game dic
elif project is 'f':
    if data_base is 'vr':
        game_dic = game_dic_all
    elif data_base is 'vr_new':
        game_dic = game_dic_new_all
    if debugging is True:
        game_dic = game_dic[debugging_range[0]:debugging_range[1]]

'''default config'''

if (project is 'f') and (mode is 'on_line'):
    if data_base is 'vr':
        num_subjects = 40
    elif data_base is 'vr_new':
        num_subjects = 58
    worker_done_signal_dir = 'temp/worker_done_signal_dir/'
    worker_done_signal_file = 'worker_done_signal.npz'
    check_worker_done_time = 2# 60 * 10
else:
    num_games_global = len(game_dic)
    num_workers_global = 16
    num_workers_total_global = num_games_global * num_workers_global
    task_plus = cluster_current * num_workers_total_global
    task_chief = cluster_main * num_workers_total_global
    task_plus = cluster_current * num_workers_total_global
    task_chief = cluster_main * num_workers_total_global

if project is 'g':
    game_dic_all = get_env_dic([
        'alien', 'amidar', 'bank_heist', 'ms_pacman', 'tutankham', 'venture', 'wizard_of_wor', # maze >> g7s0
        'assault', 'asteroids', 'beam_rider', 'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'atlantis', 'gravitar', 'phoenix', 'pooyan', 'riverraid', 'seaquest', 'space_invaders', 'star_gunner', 'time_pilot', 'zaxxon', 'yars_revenge', # shot 3 >> g18s7
        'asterix', 'elevator_action', 'berzerk', 'freeway', 'frostbite', 'journey_escape', 'kangaroo', 'krull', 'pitfall', 'skiing', 'up_n_down', 'qbert', 'road_runner', # advanture >> g13s25
        'double_dunk', 'ice_hockey', 'montezuma_revenge', 'gopher', # iq >> g4s38
        'breakout', 'pong', 'private_eye', 'tennis', 'video_pinball', # pong >> g5s42
        'fishing_derby', 'name_this_game', # fishing >> g2s47
        'bowling', # bowing >> g1s49
        'battle_zone', 'boxing', 'jamesbond', 'robotank', 'solaris', # shot 1 >> g5s50
        'enduro', # drive 1 >> g1s55
        'kung_fu_master', # fight >> g1s56
    ])
    game_dic_all_ac_space = get_env_dic_ac_space(game_dic_all)
elif project is 'f':
    my_sigma = (11.75+13.78)/2
    import math
    sigma_half_fov = 51.0 / (math.sqrt(-2.0*math.log(0.5)))

games_start_global = 0

cluster_host = ['192.168.226.67', '192.168.226.27', '192.168.226.139', '192.168.1.31'] # main cluster has to be first
cluster_name = ['yuhangsong'    , 'server'        , 'worker',          'xuntian2'] # main cluster has to be first
cluster_home = ['yuhangsong'    , 's'             , 'irc207',          'xuntian2'] # main cluster has to be first

'''compute_consi config, only works in mode data_processor'''
if project is 'f' and mode is 'data_processor':
    fov_degree = 6
    no_moving_gate = 0.0001
    compute_lon_inter = fov_degree / 2
    compute_lat_inter = fov_degree / 2
    frame_gate = 20
    MaxCenterNum = 4
    NumDirectionForCluster = 8
    DirectionInter = 360 / NumDirectionForCluster
