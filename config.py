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
status = ""
basic_log_dir = "gtn_3"
log_dir = "test_gtn_1"

cluster_current = 0 # specific current cluster here
cluster_main = 0

if_restore_model = False
if if_restore_model is True:
    model_to_restore = "../../result/model_to_restore/model.ckpt-8496809"

'''model structure'''
consi_depth = 3
lstm_size = [288,128,32]

if_mix_exp = False
if_reward_auto_normalize = False

num_workers_global = 4

update_step = 20

game_dic_test_single_pong = [
    'pong',
]
game_dic_test_multi_pong = [
    'pong', 'breakout',
]
game_dic_shooting = [
    'assault', 'asteroids', 'beam_rider', 'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'atlantis', 'gravitar', 'phoenix', 'pooyan', 'riverraid', 'seaquest', 'space_invaders', 'star_gunner', 'time_pilot', 'zaxxon', 'yars_revenge',
]
game_dic = game_dic_test_multi_pong

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
def get_env_seq(env_seq_id):
    import copy
    env_seq_id = copy.deepcopy(env_seq_id)
    print("Total Games:" + str(len(env_seq_id)))
    for i in range(len(env_seq_id)):
        name = ''.join([g.capitalize() for g in env_seq_id[i].split('_')])
        env_seq_id[i] = '{}Deterministic-v3'.format(name)
    return env_seq_id
def get_env_ac_space(env_id):
    import envs
    return envs.create_atari_env(env_id).action_space.n
'''
for env_id_i in config.get_env_seq(config.game_dic_all):
    self.ac[env_id_i] = tf.placeholder(tf.float32, [None, envs.create_atari_env(env_id_i).action_space.n], name="ac_"+env_id_i)
    self.adv[env_id_i] = tf.placeholder(tf.float32, [None], name="adv_"+env_id_i)
    self.r[env_id_i] = tf.placeholder(tf.float32, [None], name="r_"+env_id_i)
    self.step_forward[env_id_i] = tf.placeholder(tf.int32, [None], name="step_forward_"+env_id_i)

    log_prob_tf[env_id_i] = tf.nn.log_softmax(pi.logits[env_id_i])
    prob_tf[env_id_i] = tf.nn.softmax(pi.logits[env_id_i])

    # the "policy gradients" loss:  its derivative is precisely the policy gradient
    # notice that self.ac is a placeholder that is provided externally.
    # ac will contain the advantages, as calculated in process_rollout
    pi_loss[env_id_i] = - tf.reduce_sum(tf.reduce_sum(log_prob_tf[env_id_i] * self.ac[env_id_i], [1]) * self.adv[env_id_i])

    # loss of value function
    vf_loss[env_id_i] = 0.5 * tf.reduce_sum(tf.square(pi.vf[env_id_i] - self.r[env_id_i]))
    entropy[env_id_i] = - tf.reduce_sum(prob_tf[env_id_i] * log_prob_tf[env_id_i])


    self.loss[env_id_i] = pi_loss[env_id_i] + 0.5 * vf_loss[env_id_i] - entropy[env_id_i] * 0.01
'''
