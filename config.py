'''for cluster'''
'''
    if restore:
        with main cluster:
            put model to ../../result/model_to_restore/
            set if_restore_model to True
        if continou log:
            set log_dir_global
    modify cluster_current only
    run main cluster first
    then run other

'''

status = "coding"
basic_log_dir = "gtn_1"
log_dir = "test_9"

cluster_current = 0 # specific current cluster here
cluster_main = 0

if_restore_model = False
if if_restore_model is True:
    model_to_restore = "../../result/model_to_restore/model.ckpt-8496809"

if_mix_exp = True
if_reward_auto_normalize = False

num_workers_global = 8

update_step = 20

game_dic_test_single_pong = [
    'pong',
]
game_dic_test_multi_pong = [
    'pong', 'breakout',
]
game_dic = game_dic_test_single_pong

'''default'''
num_games_global = len(game_dic)
num_workers_total_global = num_games_global * num_workers_global
game_dic_all = [
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
]
games_start_global = 0
mix_exp_temp_dir = 'mix_exp_temp_dir/'
cluster_host = ['192.168.226.67', '192.168.226.27', '192.168.226.139'] # main cluster has to be first
cluster_name = ['yuhangsong'    , 'server'        , 'worker'] # main cluster has to be first
cluster_home = ['yuhangsong'    , 's'             , 'irc207'] # main cluster has to be first
