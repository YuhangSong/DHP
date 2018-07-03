import argparse
import os
import sys
import gym
import config
import copy
import time
import numpy as np
import subprocess

def run():

    session = "a3c"

    cmds = [
        "tmux kill-session -t {}".format(session),
    ]
    '''excute cmds'''
    os.system("\n".join(cmds))

if __name__ == "__main__":
    run()
