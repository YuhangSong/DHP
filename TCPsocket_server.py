#!/usr/bin/env python
#
# -*- coding:utf-8 -*-
#

from socket import *
from time import ctime

HOST = ''
PORT = 21567
BUFSIZE=1024
ADDR=(HOST, PORT)

tcpSrvSock=socket(AF_INET, SOCK_STREAM)
tcpSrvSock.bind(ADDR)
tcpSrvSock.listen(5)

while True:
    print 'waiting for connection ...'
    tcpCliSock,addr = tcpSrvSock.accept()
    print '... connected from:', addr

    while True:
        data=tcpCliSock.recv(BUFSIZE)
        if not data:
            break
        tcpCliSock.send('[%s] %s'%(ctime(), data))
        print [ctime()],':',data

tcpCliSock.close()
tcpSrvSock.close()
