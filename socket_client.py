#!/usr/bin/env python
import socket
import struct
import time
import numpy as np
from numpy import array
from socket_server import AA, PushBatch
try:
    import cPickle as pickle
except:
    import pickle

if __name__ == '__main__':

    '''connection name'''
    conn = '/tmp/conn'

    '''create the socket'''
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    '''connect to the socket'''
    sock.connect(conn)

    send_batch = PushBatch()
    send_batch.data[1][1][1][0]=1.231
    send_stream = pickle.dumps(send_batch)
    sock.sendall(send_stream)

    recieve_stream=sock.recv(1024)
    recieve_batch=pickle.loads(recieve_stream)
    print(np.shape(recieve_batch.get_data()))
    print(recieve_batch.get_data()[1][1][1][0])
    sock.close()
