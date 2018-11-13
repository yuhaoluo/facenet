#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 12:25:54 2018

@author: luoyuhao
"""
import base64
import cv2
import numpy as np
import imageio
from scipy import misc
import json
import time
import os
import sklearn
from vector_normalization import Vector
def base64_to_image(base64_code):
    # base64解码
    img_data = base64.b64decode(base64_code)
    # 转换为np数组
    img_array = np.frombuffer(img_data, np.uint8)
    # 转换成opencv可用格式
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def image_to_base64(image_np):

    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image = cv2.imencode('.jpg',image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code
##########################################################
## named by date time
def save_faces_to_docker(res,docker_folder):
    if not os.path.exists(docker_folder):
        os.makedirs(docker_folder)
    date_str = Times(time)
    filename = date_str+'.png'
    output_filename = os.path.join(docker_folder,filename)
    if(len(res)>0):
        filename_base, file_extension = os.path.splitext(output_filename)
    for i in range(len(res)):
        if (len(res)>1):
            output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
        else:
            output_filename_n = "{}{}".format(filename_base, file_extension)
        imageio.imwrite(output_filename_n, res[i])
        
##################################################################
## named by idx
def save_faces_to_docker2(res,docker_folder,cam_id,cap_ts,index=1):
    if not os.path.exists(docker_folder):
        os.makedirs(docker_folder)
    for i in range(len(res)):

        filename = str(cam_id)+'-'+str(cap_ts)+'-'+str(index)+'.jpg'
        output_filename = os.path.join(docker_folder,filename)
        imageio.imwrite(output_filename,res[i])
        index = index+1
    return index

# =============================================================================
# img_path = "/home/luoyuhao/Datasets/Docker/1.png"
# img_io = imageio.imread(img_path)
# encode_img = image_to_base64(img_io)
# decode_img = base64_to_image(encode_img)
# print(decode_img.shape)
# 
# =============================================================================


# =============================================================================
# log_path = '/home/luoyuhao/ai2.log'
# ai_type = 'ai'
# ai_uuid = '8090'
# msg = 'happy'
# a = np.array([5,6])
# msg = 'read image: {} fail {} {}.'.format(log_path,a.dtype,a.shape)
# print(msg)
# =============================================================================

def Times(time):
    date_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    return date_time

def write_msg(log_path,ai_type,ai_uuid,msg,date_time,capId='',capTs=''):
    (filepath,tempfilename) = os.path.split(log_path)
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    with open(log_path,'at') as f:
        f.write('date:%s\tmodule:%s\tuuid:%s\tmsg:%s\tcapId:%s\tcapTs:%s\n' % (date_time,ai_type,ai_uuid,msg,capId,capTs)) 
        
#write_msg(log_path,ai_type,ai_uuid,msg,Times(time),34,67)

def array64_to_base64(array):
    return base64.b64encode(array)

def base64_to_array64(base64_code):
    data = base64.b64decode(base64_code)
    array = np.frombuffer(data, dtype=np.float64)
    return array


# =============================================================================
# arr = 0.5 * np.ones(256)
# print(arr.dtype)
# print(arr.shape)
# 
# 
# b64code = array64_to_base64(arr)
# decode_arr = base64_to_array64(b64code)
# 
# lib_features_path = '/home/luoyuhao/Datasets/ms-celeb-1M-evaluation/lib_features.npy'
# lib_fea = np.load(lib_features_path)
# fea1 = lib_fea[0]
# 
# fea1_64code = array64_to_base64(fea1)
# fea1_decode = base64_to_array64(fea1_64code)
# 
# print("decode res:")
# print(np.allclose(arr,decode_arr ))
# print(np.allclose(fea1,fea1_decode))
# =============================================================================

def read_labels_and_features_from_json(path):
    #path="/home/luoyuhao/Datasets/ai_cloud_evaluate/json_lib/whiteList_256_20181102_1.json"
    f=open(path, "r")
    dicts = json.load(f)
    labelfeaturelists=dicts["labelfeatures"]
    labels  = []
    features = []
    for every in labelfeaturelists:
        labels.append((every["label"]))
        features.append(np.array(every["feature"]))
    return labels, features	

def create_face_lib(labels,features):
    lib_dic = []
    if len(labels)!=len(features):
        print('the length of labels  do not match  with features size.')
        return lib_dic
    else:
        for i in range(len(labels)):
            item = {'feature':features[i],'label':labels[i]}
            lib_dic.append(item)
        print("face lib size:", len(lib_dic))
        return lib_dic
        
 
# =============================================================================
# face_json_path = "/home/luoyuhao/Datasets/ai_cloud_evaluate/json_lib/whiteList_256_20181102_1.json" 
# start = time.time()      
# labels,features = read_labels_and_features_from_json(face_json_path)
# faces_lib_dic = create_face_lib(labels,features)
# print('time:',time.time()-start)
# 
# 
# img_path = '/home/luoyuhao/Documents/Code/download_from_github/Face_Verification/vgg_face_caffe/2/b1.png'
# save_path = '/home/luoyuhao/Documents/Code/download_from_github/Face_Verification/vgg_face_caffe/b1_224.png'
# img_io = imageio.imread(img_path)
# img_resize = misc.imresize(img_io,(224, 224), interp='bilinear')
# imageio.imwrite(save_path,img_resize)
# =============================================================================

def MaxMinNormalization(x,Max,Min):
	x = (x - Min) / (Max - Min);
	return x;


##  mu:np.average(), sigma: np.std()
def Z_ScoreNormalization(x,mu,sigma):
	x = (x - mu) / sigma;
	return x;






arr = np.array([30.2946,65.5318,48.0252,33.5493,62.7299,51.6963,51.5014,71.7366,92.3655,92.2041])
point_x = arr[0:5]
point_y = arr[5:10]
arr2 = np.zeros((5,2))
arr2[:,0] = point_x
arr2[:,1] = point_y
print(arr2)

