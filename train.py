import argparse
import os
import sys
import gym
from envs import if_training
from config import games_start_global, num_games_global, num_workers_global, num_workers_total_global
from config import cluster_current, cluster_main, log_dir_global, status

parser = argparse.ArgumentParser(description="Run commands")

task_plus = cluster_current * num_workers_total_global

'''
specific exp name for different run
'''
parser.add_argument('-x', '--exp-id', type=str, default=log_dir_global + status,
                    help="Experiment id")

'''
hyper paramters
'''
parser.add_argument('-d', '--consi-depth', type=int, default=1,
                    help="hyper paramter: depth of the consciousness")

'''
max to 50 games
'''
parser.add_argument('-g', '--num-games', default=num_games_global, type=int,
                    help="Number of games")
parser.add_argument('-s', '--games-start', default=games_start_global, type=int,
                    help="Games start position")
parser.add_argument('-w', '--num-workers', default=num_workers_global, type=int,
                    help="Number of workers")

'''
normoly default paramters
'''
parser.add_argument('-r', '--remotes', default=None,
                    help='The address of pre-existing VNC servers and '
                         'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).')
parser.add_argument('-v', '--game-version', type=str, default="Deterministic-v3",
                    help="Game version, not in usage")
parser.add_argument('-l', '--log-dir', type=str, default="../../result/ff40"+status+"/" + parser.parse_args().exp_id,
                    help="Log directory path")

def get_env_seq():

    env_seq_id = [
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
        'pong', 'breakout', #test >> g2s57
        'ff' #ff >> g1s59
    ]

    print("Total Games:" + str(len(env_seq_id)))

    for i in range(len(env_seq_id)):
        name = ''.join([g.capitalize() for g in env_seq_id[i].split('_')])
        env_seq_id[i] = '{}Deterministic-v3'.format(name)

    return env_seq_id

def get_env_seq_ff():

    if(if_training==True):
        '''specific training set'''
        env_seq_id = [
            'Pokemon',
            'Gliding',
            'Parachuting',
            'RollerCoaster',
            'Skiing',
            'CS',
            'Dota2',
            'GalaxyOnFire',
            'LOL',
            'MC',
            'BTSRun',
            'Graffiti',
            'KasabianLive',
            'LetsNotBeAloneTonight',
            'Antarctic',
            'BlueWorld',
            'Dubai',
            'Egypt',
            'StarryPolar',
            'A380',
            'CandyCarnival',
            'MercedesBenz',
            'RingMan',
            'RioOlympics',
            'Help',
            'IRobot',
            'Predator',
            'ProjectSoul',
            'AirShow',
            'DrivingInAlps',
            'F5Fighter',
            'HondaF1',
            'Rally',
            'AcerPredator',
            'BFG',
            'CMLauncher',
            'Cryogenian',
        ]
    else:
        '''specific test set'''
        env_seq_id = [
            'Surfing',
            'Waterskiing',
            'SuperMario64',
            'Symphony',
            'WaitingForLove',
            'WesternSichuan',
            'VRBasketball',
            'StarWars',
            'Terminator',
            'Supercar',
            'LoopUniverse',
        ]

    print("Total Games ff:" + str(len(env_seq_id)))

    return env_seq_id

def new_tmux_cmd(session, name, cmd):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(str(v) for v in cmd)
    return name, "tmux send-keys -t {}:{} '{}' Enter".format(session, name, cmd)


def create_tmux_commands(session, consi_depth, num_workers_per_game, num_games, remotes, logdir, game_version, games_start):

    '''
    Coder: YuhangSong
    Description: specific sequence of games to run
    '''
    env_seq_id = get_env_seq()
    env_seq_id_ff = get_env_seq_ff()

    num_workers = num_workers_per_game * num_games
    # for launching the TF workers and for launching tensorboard
    base_cmd = [
        'CUDA_VISIBLE_DEVICES=', sys.executable, 'worker.py',
        '--log-dir', logdir, '--env-id', env_seq_id[0], '--id-ff', env_seq_id_ff[0],
        '--num-workers', str(num_workers)]

    if remotes is None:
        remotes = ["1"] * num_workers
    else:
        remotes = remotes.split(',')
        assert len(remotes) == num_workers

    if(cluster_current==cluster_main):
        cmds_map = [new_tmux_cmd(session, "ps", base_cmd + ["--job-name", "ps"])]
    else:
        cmds_map = []
    for i in range(num_workers):
        if((i % num_workers_per_game)==0):
            if_log = True
        else:
            if_log = False
        base_cmd = [
            'CUDA_VISIBLE_DEVICES=', sys.executable, 'worker.py',
            '--log-dir', logdir,
            '--env-id', env_seq_id[59],
            '--id-ff', env_seq_id_ff[i / num_workers_per_game + games_start],
            '--select', str(i),
            '--num-workers', str(num_workers),
            '--if-log', str(if_log)]
        cmds_map += [new_tmux_cmd(session,
                                  "w-%d" % i,
                                  base_cmd + ["--job-name", "worker",
                                              "--task", str(i+task_plus),
                                              "--consi-depth", consi_depth,
                                              "--remotes", remotes[i]])]

    # cmds_map += [new_tmux_cmd(session, "tb", ["tensorboard --logdir {} --port 12345".format(logdir)])]
    # cmds_map += [new_tmux_cmd(session, "htop", ["htop"])]
    # if(if_mix_exp==True):
    #     cmds_map += [new_tmux_cmd(session, "experience_server", ["python experience_server.py"])]

    windows = [v[0] for v in cmds_map]

    cmds = [
        "mkdir -p {}".format(logdir),
        "tmux kill-session -t {}".format(session),
        "tmux new-session -s {} -n {} -d".format(session, windows[0]),
    ]
    for w in windows[1:]:
        cmds += ["tmux new-window -t {} -n {}".format(session, w)]
    cmds += ["sleep 1"]
    for window, cmd in cmds_map:
        cmds += [cmd]

    return cmds


def run():
    args = parser.parse_args()

    cmds = create_tmux_commands("a3c", args.consi_depth, args.num_workers, args.num_games, args.remotes, args.log_dir, args.game_version, args.games_start)
    print("\n".join(cmds))
    os.system("\n".join(cmds))

if __name__ == "__main__":
    run()
