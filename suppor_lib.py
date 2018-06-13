import cv2
import numpy as np
from math import radians, cos, sin, asin, sqrt, log
import math
import copy
import scipy
import config

def calc_angle(x_point_s,y_point_s,x_point_e,y_point_e):

    '''
    Function: compute direction to north from Mercator cordication
    Coder: syh
    Status: unchecked
    '''

    angle=0
    y_se= y_point_e-y_point_s;
    x_se= x_point_e-x_point_s;
    if x_se==0 and y_se>0:
        angle = 360
    if x_se==0 and y_se<0:
        angle = 180
    if y_se==0 and x_se>0:
        angle = 90
    if y_se==0 and x_se<0:
        angle = 270
    if x_se>0 and y_se>0:
       angle = math.atan(x_se/y_se)*180/math.pi
    elif x_se<0 and y_se>0:
       angle = 360 + math.atan(x_se/y_se)*180/math.pi
    elif x_se<0 and y_se<0:
       angle = 180 + math.atan(x_se/y_se)*180/math.pi
    elif x_se>0 and y_se<0:
       angle = 180 + math.atan(x_se/y_se)*180/math.pi
    return angle

class subject():

    '''
    Function: data structure
    Coder: syh
    Status: coding
    '''

    def __init__(self, num_data_frame):
        self.data_frame = list(data_frame() for i in range(num_data_frame))

class data_frame():

    '''
    Function: data structure
    Coder: syh
    Status: coding
    '''

    def __init__(self):
        self.p = [0.0, 0.0] #position
        self.theta = 0.0 # direction
        self.v = 0.0 # speed
        self.t = 1.0 # trustworthy for position
        self.g = 1.0 # trustworthy for direction

def get_num_subjects(data):
    num_subject = np.shape(data)[1] / 2
    return num_subject

def get_subjects(data, subject_id=0):

    '''get essiancial config'''
    num_subject = np.shape(data)[1] / 2
    num_data_frame = np.shape(data)[0]

    '''subjects'''
    subjects = list(subject(num_data_frame=num_data_frame) for i in range(num_subject))

    '''set in data from mat to structure'''
    for subject_i in range(num_subject):

        for data_frame_i in range(num_data_frame):

            subjects[subject_i].data_frame[data_frame_i].p[0] = data[data_frame_i, subject_i*2 + 1] #lon
            subjects[subject_i].data_frame[data_frame_i].p[1] = data[data_frame_i, subject_i*2] #lat
            if subjects[subject_i].data_frame[data_frame_i].p[0] < -180.0 or subjects[subject_i].data_frame[data_frame_i].p[0] > 180.0:
                print(j)

        for data_frame_i in range(1, num_data_frame - 1):

            last_frame = subjects[subject_i].data_frame[data_frame_i - 1]
            next_frame = subjects[subject_i].data_frame[data_frame_i + 1]
            v = haversine(lon1=last_frame.p[0],
                          lat1=last_frame.p[1],
                          lon2=next_frame.p[0],
                          lat2=next_frame.p[1]) / 2.0
            l_x, l_y = lonlat2Mercator(lon=last_frame.p[0],
                                       lat=last_frame.p[1])
            n_x, n_y = lonlat2Mercator(lon=next_frame.p[0],
                                       lat=next_frame.p[1])
            theta = calc_angle(x_point_s=l_x,
                               y_point_s=l_y,
                               x_point_e=n_x,
                               y_point_e=n_y)

            subjects[subject_i].data_frame[data_frame_i].v = v
            subjects[subject_i].data_frame[data_frame_i].theta = theta

            if(subject_i==1):
                subjects[subject_i].data_frame[0].v = v
                subjects[subject_i].data_frame[0].theta = theta
            if(subject_i==(num_data_frame - 2)):
                subjects[subject_i].data_frame[num_data_frame - 1].v = v
                subjects[subject_i].data_frame[num_data_frame - 1].theta = theta

    return num_subject, num_data_frame, subjects, subjects[subject_id]

def get_transfered_data(lon, lat, theta, data_frame, max_distance_on_position=1.0*math.pi, max_distance_on_degree=180.0, final_discount_to=10**(-2)):

    distance_on_position = haversine(lon1=lon,
                                     lat1=lat,
                                     lon2=data_frame.p[0],
                                     lat2=data_frame.p[1])

    distance_on_degree = abs(theta - data_frame.theta)
    if(distance_on_degree>180):
        distance_on_degree = distance_on_degree - 180

    thegma_2_on_position = -0.5*(max_distance_on_position**2)/math.log(final_discount_to)
    thegma_2_on_degree = -0.5*(max_distance_on_degree**2)/math.log(final_discount_to)

    '''guassion trustworthy transfer'''
    prob = 1.0 * math.exp(-1.0 / 2.0 * (distance_on_position**2) / (thegma_2_on_position)) * math.exp(-1.0 / 2.0 * (distance_on_degree**2) / (thegma_2_on_degree))

    return prob

def get_prob(lon, lat, theta, subjects, subjects_total, cur_data):

    prob_dic = []
    for i in range(subjects_total):
        from config import final_discount_to
        x = get_transfered_data(lon=lon,
                                lat=lat,
                                theta=theta,
                                data_frame=subjects[i].data_frame[cur_data],
                                final_discount_to=final_discount_to)
        prob_dic += [x]
    '''prob will not be normalized, since we do not encourage move to unmeasured area'''
    prob_sum = sum(prob_dic)

    prob_dic = np.array(prob_dic)

    '''normalize prob dic'''
    prob_dic_normalized = prob_dic / prob_sum

    distance_per_data = 0.0
    for i in range(subjects_total):
        if config.if_normalize_v_lable:
            distance_per_data += prob_dic_normalized[i] * subjects[i].data_frame[cur_data].v
        else:
            distance_per_data += prob_dic[i] * subjects[i].data_frame[cur_data].v

    if(distance_per_data<=0):
        print("!!!! v too small !!!!")
        distance_per_data = 0.00001

    return (prob_sum/subjects_total), distance_per_data

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 1.0
    return c * r


def lonlat2Mercator(lon, lat):

    '''
    Function: convert lon-lat to Mercator cardination
    Coder: syh
    Status: unchecked
    '''

    x = lon * 20037508.34 / 180
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0

    return x, y


def fixation2salmap(fixation, mapwidth, mapheight, my_sigma_in_degree = 7, sp = True):
    my_sigma_in_degree = 7
    fixation_total = np.shape(fixation)[0]
    x_degree_per_pixel = 360.0 / mapwidth
    y_degree_per_pixel = 180.0 / mapheight
    salmap = np.zeros((mapwidth, mapheight))
    for x in range(mapwidth):
        for y in range(mapheight):
            cur_lon = - x * x_degree_per_pixel + 180.0
            cur_lat = - y * y_degree_per_pixel + 90.0
            for fixation_count in range(fixation_total):
                cur_fixation_lon = fixation[fixation_count][0]
                cur_fixation_lat = fixation[fixation_count][1]
                if sp is True:
                    distance_to_cur_fixation = haversine(lon1=cur_lon,
                                                         lat1=cur_lat,
                                                         lon2=cur_fixation_lon,
                                                         lat2=cur_fixation_lat)
                    distance_to_cur_fixation_in_degree = distance_to_cur_fixation / math.pi * 180.0
                    sal = math.exp(-1.0 / 2.0 * (distance_to_cur_fixation_in_degree**2) / (my_sigma_in_degree**2))
                else:
                    print('currently not supporting none-sp map')
                salmap[x, y] += sal
    salmap = salmap * (1.0 / np.amax(salmap))
    salmap = np.transpose(salmap)
    return salmap

def constrain_degree_to_0_360(direction):
    return (direction+360.0) if (direction<0) else (direction+0.0)
