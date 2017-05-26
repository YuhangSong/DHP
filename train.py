import argparse
import os
import sys
import gym
import config
import copy

parser = argparse.ArgumentParser(description="Run commands")

task_plus = config.cluster_current * config.num_workers_total_global


'''
normoly default paramters
'''
parser.add_argument('-r', '--remotes', default=None,
                    help='The address of pre-existing VNC servers and '
                         'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).')

def new_tmux_cmd(session, name, cmd):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(str(v) for v in cmd)
    return name, "tmux send-keys -t {}:{} '{}' Enter".format(session, name, cmd)


def create_tmux_commands(session, remotes, logdir):

    '''
    Coder: YuhangSong
    Description: specific sequence of games to run
    '''
    env_seq_id = config.get_env_seq(config.game_dic)

    # for launching the TF workers and for launching tensorboard
    base_cmd = [
        'CUDA_VISIBLE_DEVICES=', sys.executable, 'worker.py',
        '--log-dir', logdir, '--env-id', env_seq_id[0],
        '--num-workers', str(config.num_workers_total_global)]

    if remotes is None:
        remotes = ["1"] * config.num_workers_total_global
    else:
        remotes = remotes.split(',')
        assert len(remotes) == config.num_workers_total_global

    '''main cluster has ps worker'''
    if(config.cluster_current==config.cluster_main):
        cmds_map = [new_tmux_cmd(session, "ps", base_cmd + ["--job-name", "ps"])]
    else:
        cmds_map = []

    for i in range(config.num_workers_total_global):
        if((i % config.num_workers_global) >= config.num_workers_local):
            continue
        base_cmd = [
            'CUDA_VISIBLE_DEVICES=', sys.executable, 'worker.py',
            '--log-dir', logdir,
            '--env-id', env_seq_id[i / config.num_workers_global],
            '--select', str(i),
            '--num-workers', str(config.num_workers_total_global)]
        cmds_map += [new_tmux_cmd(session,
                                  "w-%d" % i,
                                  base_cmd + ["--job-name", "worker",
                                              "--task", str(i+task_plus),
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

    cmds = create_tmux_commands("a3c", args.remotes, "../../result/"+config.basic_log_dir+config.status+"/" + config.log_dir + config.status)
    print("\n".join(cmds))
    os.system("\n".join(cmds))

if __name__ == "__main__":
    run()
