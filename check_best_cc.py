import argparse
import os
import sys
import gym
import copy
import time
import numpy as np
import subprocess

def check_best_cc():

    # print('=======================checking best cc==========================')

    best_cc_dic = {}

    from config import game_dic
    for i in range(len(game_dic)):

        env_id = game_dic[i]

        from config import final_log_dir
        record_dir = final_log_dir+'ff_best_cc/'+env_id+'/'

        try:
            best_cc_dic[env_id] = np.load(record_dir+'best_cc.npz')['best_cc'][0]
        except Exception, e:
            pass
            # print(str(Exception)+": "+str(e))

    if len(best_cc_dic) is 0:
        return

    best_cc_dic=sorted(best_cc_dic.items(), key=lambda e:e[1], reverse=True)

    print('=======================sorted cc==========================')
    for i in range(len(best_cc_dic)):
        print(best_cc_dic[i][0]+'\t'+str(best_cc_dic[i][1]))
        # with open("best_cc.txt","a") as f:
        #     f.write("%s\tsubject[%s]:\t%s\n"%(self.env_id,self.subject,mo_mean))



def run():

    '''detecting'''
    while True:

        check_best_cc()

        '''sleep for we do not need to detecting very frequent'''
        from config import check_worker_done_time
        time.sleep(check_worker_done_time)


if __name__ == "__main__":
    run()
