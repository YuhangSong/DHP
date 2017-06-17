import cv2
import numpy as np
import sys
from math import radians, cos, sin, asin, sqrt, log
import math
from scipy.misc import imsave
import subprocess
from read_yuv import yuv_import
from config import cluster_name, cluster_current
if cluster_name[cluster_current] is 'xuntian2':
    import PIL.Image as Image
else:
    import Image
from numpy import *

def get_view(input_width,input_height,view_fov_x,view_fov_y,view_center_lat,view_center_lon,output_width,output_height,cur_frame,file_,is_render=False,temp_dir=""):
    temp_1=temp_dir+"1.yuv"
    import config
    subprocess.call(["/home/"+config.cluster_home[config.cluster_current]+"/remap", "-i", "rect", "-o", "view", "-m", str(input_height), "-b", str(input_width), "-w", str(output_width), "-h", str(output_height), "-x", str(view_fov_x), "-y", str(view_fov_y), "-p", str(view_center_lat), "-l", str(view_center_lon), "-z", "1", "-s", str(cur_frame), file_, temp_1])
    frame=yuv_import(temp_1,(output_height,output_width),1,0)
    subprocess.call(["rm", temp_1])

    if(is_render==True):

        print("this is debugging, not trainning")
        YY=frame[0]
        im=Image.frombytes('L',(output_height,output_width),YY.tostring())
        im.show()
        frame = np.zeros((42,42,1))
        frame = np.reshape(frame, [42, 42, 1])

    else:

        frame = np.array(frame)
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        frame = np.reshape(frame, [42, 42, 1])

    return frame
