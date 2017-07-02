'''
    Description: game dic for g
'''
def get_env_dic(env_seq_id):
    import copy
    env_seq_id = copy.deepcopy(env_seq_id)
    print("Total Games:" + str(len(env_seq_id)))
    for i in range(len(env_seq_id)):
        name = ''.join([g.capitalize() for g in env_seq_id[i].split('_')])
        env_seq_id[i] = '{}Deterministic-v3'.format(name)
    return env_seq_id
def get_env_ac_space(env_id):
    import gym
    return gym.make(env_id).action_space.n
def get_env_dic_ac_space(env_id_dic):
    env_dic_ac_space = {}
    for env_id in env_id_dic:
        env_dic_ac_space[env_id] = get_env_ac_space(env_id)
    return env_dic_ac_space
'''game dic'''
g_game_dic_test_single_pong = get_env_dic([
    'pong',
])
g_game_dic_test_multi_pong = get_env_dic([
    'pong', 'breakout',
])
g_game_dic_shooting = get_env_dic([
    'assault', 'asteroids', 'beam_rider', 'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'atlantis', 'gravitar', 'phoenix', 'pooyan', 'riverraid', 'seaquest', 'space_invaders', 'star_gunner', 'time_pilot', 'zaxxon', 'yars_revenge',
])
g_game_dic_all = get_env_dic([
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
g_game_dic_all_ac_space = get_env_dic_ac_space(g_game_dic_all)
