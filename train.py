import argparse
import os
import sys
import gym
import config

parser = argparse.ArgumentParser(description="Run commands")

task_plus = config.cluster_current * config.num_workers_total_global

'''
specific exp name for different run
'''
parser.add_argument('-x', '--exp-id', type=str, default=config.log_dir + config.status,
                    help="Experiment id")

'''
hyper paramters
'''
parser.add_argument('-d', '--consi-depth', type=int, default=1,
                    help="hyper paramter: depth of the consciousness")

'''
max to 50 games
'''
parser.add_argument('-g', '--num-games', default=config.num_games_global, type=int,
                    help="Number of games")
parser.add_argument('-s', '--games-start', default=config.games_start_global, type=int,
                    help="Games start position")
parser.add_argument('-w', '--num-workers', default=config.num_workers_global, type=int,
                    help="Number of workers")

'''
normoly default paramters
'''
parser.add_argument('-r', '--remotes', default=None,
                    help='The address of pre-existing VNC servers and '
                         'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).')
parser.add_argument('-v', '--game-version', type=str, default="Deterministic-v3",
                    help="Game version, not in usage")
parser.add_argument('-l', '--log-dir', type=str, default="../../result/"+config.basic_log_dir+config.status+"/" + parser.parse_args().exp_id,
                    help="Log directory path")

def get_env_seq():

    env_seq_id = config.game_dic

    print("Total Games:" + str(len(env_seq_id)))

    for i in range(len(env_seq_id)):
        name = ''.join([g.capitalize() for g in env_seq_id[i].split('_')])
        env_seq_id[i] = '{}Deterministic-v3'.format(name)

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

    num_workers = num_workers_per_game * num_games
    # for launching the TF workers and for launching tensorboard
    base_cmd = [
        'CUDA_VISIBLE_DEVICES=', sys.executable, 'worker.py',
        '--log-dir', logdir, '--env-id', env_seq_id[0],
        '--num-workers', str(num_workers)]

    if remotes is None:
        remotes = ["1"] * num_workers
    else:
        remotes = remotes.split(',')
        assert len(remotes) == num_workers

    if(config.cluster_current==config.cluster_main):
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
            '--env-id', env_seq_id[i / num_workers_per_game + games_start],
            '--select', str(i),
            '--num-workers', str(num_workers)]
        cmds_map += [new_tmux_cmd(session,
                                  "w-%d" % i,
                                  base_cmd + ["--job-name", "worker",
                                              "--task", str(i+task_plus),
                                              "--consi-depth", consi_depth,
                                              "--remotes", remotes[i]])]

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
