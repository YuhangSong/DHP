#!/usr/bin/env python
from socket import *
from time import ctime
import struct
import os
import numpy as np
from numpy import array
try:
    import cPickle as pickle
except:
    import pickle
import copy

from socket_struct import PushBatch, ReturnBatch
from config import num_games_global, games_start_global, num_workers_global, num_workers_total_global, max_push_bytes, experience_server_port

if __name__ == '__main__':

    client_max_num = num_workers_total_global

    '''create exp_buf'''
    exp_buf = range(num_workers_total_global)
    exp_batch_size_buf = range(num_workers_total_global)

    '''create tcpCliSock'''
    tcpSrvSock=socket(AF_INET, SOCK_STREAM)
    HOST = ''
    PORT = experience_server_port
    ADDR=(HOST, PORT)
    tcpSrvSock.bind(ADDR)
    tcpSrvSock.listen(client_max_num)

    '''server working'''
    while True:

        print('waiting for connection ...')
        tcpCliSock,addr = tcpSrvSock.accept()
        print('... connected from:', addr)

        recieve_stream = tcpCliSock.recv(max_push_bytes) #2
        tcpCliSock.send('g') #3

        task, batch_size = recieve_stream.split('/')
        task = int(task)
        batch_size = int(batch_size)
        print('\trecieving from task\t'+str(task)+'\tbatch_size\t'+str(batch_size))
        recieve_stream = tcpCliSock.recv(max_push_bytes) #6
        print(len(recieve_stream))
        print(e)
        exp_buf[task] = recieve_stream
        exp_batch_size_buf[task] = batch_size

        task = 0

        tcpCliSock.sendall(str(task)+'/'+str(exp_batch_size_buf[task])) #7
        tcpCliSock.recv(max_push_bytes) #10

        tcpCliSock.sendall(exp_buf[task]) #11
        tcpCliSock.recv(max_push_bytes) #14

        print(d)

        '''send back mixed experience'''
        for i in range(num_workers_total_global):
            if(i==recieve_batch.id):
                continue
            else:
                '''compress to send_stream'''
                send_stream = pickle.dumps(exp_buf[i])
                '''send'''
                tcpCliSock.sendall(send_stream)
                '''waiting confirm'''
                tcpCliSock.recv(max_push_bytes)
                print("\treturn batch sized:\t"+ str(exp_buf[i].batch_size) + "\tat step:\t" + str(i))

        '''close'''
        tcpCliSock.close()

        print('============================================================')
