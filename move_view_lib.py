import math
PI=3.1415927
from math import sqrt


def trans_deg2rar(angle):       # a function to convert degree to radian
    return angle*PI/180.0

def trans_rar2deg(radian_angle):   #a function to convert radian to degree
    return radian_angle*180.0/PI

def get_car_coo(lon,lat):          #using radian
    x=math.cos(lon)*math.cos(lat)
    y=math.cos(lat)*math.sin(lon)
    z=math.sin(lat)
    return x,y,z

def get_sph_cor(x,y,z):    #using radian
    lon=0
    if(x==0 and y>0):
        lon=PI/2
    elif(x==0 and y<0):
        lon=3*PI/2
    elif(x==0 and y==0):
        print ("error")
        return
    elif(x>0 and y==0):
        lon=0
    elif(x>0 and y>0):
        lon=math.atan(float(y)/float(x))
    elif(x>0 and y<0):
        lon=2*PI+math.atan(float(y)/float(x))
    elif(x<0 and y==0):
        lon=PI
    elif(x<0 and y>0):
        lon=PI+math.atan(float(y)/float(x))
    elif(x<0 and y<0):
        lon=PI+math.atan(float(y)/float(x))

    lat=PI/2-math.acos(z/1.0)

    return lon,lat



def get_relative_sph_cor(direction,degree_per_move):    #using radian
    lon=0
    lat=0
    lon_move=math.atan(math.tan(trans_deg2rar(degree_per_move))/sqrt(2))
    lat_move=math.atan(math.tan(trans_deg2rar(degree_per_move))/sqrt(2+math.tan(trans_deg2rar(degree_per_move))**2))

    if not degree_per_move>0:
        print("wrong")
        return

    if(direction==0):
        lon=0
        lat+=trans_deg2rar(degree_per_move)
        return lon,lat
    elif(direction==1):
        lon+=2*PI-lon_move
        lat+=lat_move
        return lon,lat
    elif(direction==2):
        lon+=2*PI-trans_deg2rar(degree_per_move)
        lat=0
        return lon,lat
    elif(direction==3):
        lon+=2*PI-lon_move
        lat-=lat_move
        return lon,lat
    elif(direction==4):
        lon=0
        lat-=trans_deg2rar(degree_per_move)
        return lon,lat
    elif(direction==5):
        lon+=lon_move
        lat-=lat_move
        return lon,lat
    elif(direction==6):
        lon+=trans_deg2rar(degree_per_move)
        lat=0
        return lon,lat
    elif(direction==7):
        lon+=lon_move
        lat+=lat_move
        return lon,lat
    elif(direction==8):
        lon=0
        lat=0
        return lon,lat



def generate_vector(x,y,z):
    vector=[[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]]
    if(z==0):
        vector3_x=0
        vector3_y=0
        vector3_z=1
    elif(z<0):
        vector3_x=x
        vector3_y=y
        vector3_z=-((x*x+y*y)/z)
    elif(z>0):
        vector3_x = -x
        vector3_y = -y
        vector3_z = ((x * x + y * y) / z)
    vector3_real=sqrt(vector3_x**2+vector3_y**2+vector3_z**2)
    vector[0][0] = x
    vector[0][1] = y
    vector[0][2]=z
    vector[1][0]=- y / sqrt(x * x + y * y)
    vector[1][1]=x / sqrt(x * x + y * y)
    vector[1][2]=0
    vector[2][0]=vector3_x/vector3_real
    vector[2][1]=vector3_y/vector3_real
    vector[2][2]=vector3_z/vector3_real
    return vector




def get_absolute_car_coo(x,y,z,vector):
    x_real=x*vector[0][0]+y*vector[1][0]+z*vector[2][0]
    y_real=x*vector[0][1]+y*vector[1][1]+z*vector[2][1]
    z_real=x*vector[0][2]+y*vector[1][2]+z*vector[2][2]
    return x_real,y_real,z_real




def move_view(cur_lon, cur_lat, direction, degree_per_step):   #default R equals 1,don't change it
    if (cur_lat == 90.0):
        cur_lat = 89.9999999999999
    if (cur_lat == -90.0):
        cur_lat = -89.9999999999999
    relative_lon,relative_lat=get_relative_sph_cor(direction,degree_per_step)
    relative_x,relative_y,relative_z=get_car_coo(relative_lon,relative_lat)
    cur_x,cur_y,cur_z=get_car_coo(trans_deg2rar(cur_lon),trans_deg2rar(cur_lat))
    vector=generate_vector(cur_x,cur_y,cur_z)
    absolute_x,absolute_y,absolute_z=get_absolute_car_coo(relative_x,relative_y,relative_z,vector)
    new_lon,new_lat=get_sph_cor(absolute_x,absolute_y,absolute_z)
    new_lon= trans_rar2deg(new_lon)
    new_lat= trans_rar2deg(new_lat)

    return new_lon,new_lat
