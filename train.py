import argparse
import os
import sys
import gym
import config
import copy

parser = argparse.ArgumentParser(description="Run commands")

def new_tmux_cmd(session, name, cmd):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(str(v) for v in cmd)
    return name, "tmux send-keys -t {}:{} '{}' Enter".format(session, name, cmd)


def create_tmux_commands(session, logdir):

    '''
    Coder: YuhangSong
    Description: specific sequence of games to run
    '''
    env_seq_id = config.game_dic

    # for launching the TF workers and for launching tensorboard
    from config import project, mode
    if (project is 'g') or ( (project is 'f') and ( (mode is 'off_line') or (mode is 'data_processor') ) ):
        base_cmd = [
            'CUDA_VISIBLE_DEVICES=', sys.executable, 'worker.py',
            '--log-dir', logdir, '--env-id', env_seq_id[0],
            '--num-workers', str(config.num_workers_total_global)]

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
                '--num-workers', str(config.num_workers_total_global)]
            cmds_map += [new_tmux_cmd(session,
                                      "w-%d" % i,
                                      base_cmd + ["--job-name", "worker",
                                                  "--task", str(i+config.task_plus)])]

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
    elif (project is 'f') and (mode is 'on_line'):
        worker_running = 0
        cmds_map = []
        for game_i in range(len(env_seq_id)):
            for subjects_i in range(config.num_subjects):
                print(subjects_i)
                base_cmd = [
                    'CUDA_VISIBLE_DEVICES=', sys.executable, 'worker.py',
                    '--log-dir', logdir, '--env-id', env_seq_id[game_i],
                    '--num-workers', str(1)]

                cmds_map += [new_tmux_cmd(session, 'g-'+str(game_i)+'-s-'+str(subjects_i)+'-ps', base_cmd + ["--job-name", "ps",
                                                                                                             "--subject", str(subjects_i)])]

                base_cmd = [
                    'CUDA_VISIBLE_DEVICES=', sys.executable, 'worker.py',
                    '--log-dir', logdir,
                    '--env-id', env_seq_id[game_i],
                    '--num-workers', str(1)]
                cmds_map += [new_tmux_cmd(session,
                                          'g-'+str(game_i)+'-s-'+str(subjects_i)+'-w-0',
                                          base_cmd + ["--job-name", "worker",
                                                      "--task", str(0),
                                                      "--subject", str(subjects_i)])]
                worker_running += 1
                breakout = False
                if worker_running >= config.num_workers_one_run:
                    breakout = True
                    break
            if breakout:
                break

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

    cmds = create_tmux_commands("a3c", config.final_log_dir)
    print("\n".join(cmds))
    os.system("\n".join(cmds))

if __name__ == "__main__":
    run()
