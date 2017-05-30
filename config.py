'''g of f'''
project = 'f'

'''log config'''
status = ""
basic_log_dir = project+"_1"
log_dir = "12"
final_log_dir = "../../result/"+basic_log_dir+status+"/" + log_dir + status+'/'

# '''restore model config'''
# if_restore_model = False
# if if_restore_model is True:
#     model_to_restore = "../../result/model_to_restore/model.ckpt-8496809"

'''cluster config'''
cluster_current = 0
cluster_main = 0

'''worker config'''
num_workers_global = 16
num_workers_local = 16 # how many workers can this cluster run, DO NOT exceed num_workers_global

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
    '''for env behaivour'''
    if_log_scan_path = False
    if_log_cc = True
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
    game_dic_test_pokemon=[
        'Pokemon',
    ]

'''env config'''
if project is 'g':
    game_dic = game_dic_test_multi_pong # specific game dic
elif project is 'f':
    game_dic = game_dic_test_pokemon # specific game dic

'''default config'''
num_games_global = len(game_dic)
num_workers_total_global = num_games_global * num_workers_global
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
games_start_global = 0
cluster_host = ['192.168.226.67', '192.168.226.27', '192.168.226.139'] # main cluster has to be first
cluster_name = ['yuhangsong'    , 'server'        , 'worker'] # main cluster has to be first
cluster_home = ['yuhangsong'    , 's'             , 'irc207'] # main cluster has to be first
task_plus = cluster_current * num_workers_total_global
task_chief = cluster_main * num_workers_total_global
