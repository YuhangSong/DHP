#!/usr/bin/env python
import socket
import struct
import os
import numpy as np
from numpy import array
try:
    import cPickle as pickle
except:
    import pickle

class PushBatch():
    def __init__(self):
        self.data = [[[[1.0]*1]*42]*42]*20
        self.batch_size = 3
    def get_data(self):
        return self.data
    def get_batch_size(self):
        return self.batch_size

class AA():
    def a(self):
        print "123"

if __name__ == '__main__':
    client_max_num = 5

    '''see if the connection name exists'''
    conn = '/tmp/conn'
    if not os.path.exists(conn):
        os.mknod(conn)
    if os.path.exists(conn):
        os.unlink(conn)

    '''create the socket'''
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    '''set socket to the conn name'''
    sock.bind(conn)
    '''set mac client number'''
    sock.listen(client_max_num)

    '''server working'''
    while True:
        connection, address = sock.accept()
        recieved_data = connection.recv(1024)
        recieved_data=pickle.loads(recieved_data)
        print(np.shape(recieved_data.get_data()))
        print(recieved_data.get_data()[1][1][1][0])
        # if data == "hello,server":
        #     print "the client said:%s!\n" % data
        #     connection.send("hello,client")
        send_batch = PushBatch()
        send_batch.data[1][1][1][0]=1.22313213
        send_stream = pickle.dumps(send_batch)
        connection.sendall(send_stream)
        connection.close()
