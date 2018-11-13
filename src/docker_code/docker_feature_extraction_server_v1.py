# -*- coding: utf-8 -*-
"""
Exports the embeddings and labels of a directory of images as numpy arrays.
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from scipy import misc
import tensorflow as tf
import numpy as np
import os
import argparse
import facenet
import math
import base64
import cv2
import json
import requests
import imageio
import heapq

ai_type = 'f5eCalc'
ai_uuid = ''
ai_status = -1
config_path = '/data/common/common.json'
job_path = '/data/job/job.json'
log_path = ''



def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return dist

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
  
def load_data(image_paths, image_size,do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        images[i,:,:,:] = img
    return images

def prewhiten_img_to_tensor(decode_img,image_size):
   
    images = np.zeros((1, image_size, image_size, 3))
    images[0,:,:,:] = prewhiten(decode_img)
   
    return images

# read common.json
def read_config(config_path):
    #TODO
    dic={}
    try:
        f = open(config_path)     
        json_read = f.read()
        dic = json.loads(json_read)
        input_url = dic["input"]
        output_url = dic['output']
        #jobpath = dic['job']
        jobpath = "/data/job/job.json"
        f.close()
        uuid,state = read_job(jobpath)
        logs_path = '/data/logs/'+str(ai_uuid)+'.logs'
        return input_url,output_url,logs_path,uuid
    except Exception as e:
        print(e)
        return 
       
# read job.json    
def read_job(job_path):
    try:
        f = open(job_path)     
        json_read = f.read()
        dic = json.loads(json_read)
        uuid = dic['Uuid']
        state = dic['run']
        f.close()
        return uuid,state
    except Exception as e:
        print(e)
        return 
    

def read_state(job_path):
    #TODO
    try:
        f = open(job_path)     
        json_read = f.read()
        dic = json.loads(json_read)
        state = dic['run']
        f.close()
    except Exception as e:
        print(e)
    
    if state == 'true':
        return True
    else:
        return False

def read_input(input_url):
    #TODO
    r =[]
    taskJson_dic = []
    errorCode = []
    errorMsg = []  
    try:
        r = requests.get(input_url)
        res_dic = r.json()   #dic
    except Exception as e:
        #errorMessage = '{}: {}'.format(input_url, e)
        print("f5eCalc request get error.")
    
    try:
        errorCode = res_dic['errorCode']
        errorMsg = res_dic['errorMsg']
        if errorCode == 0:
             taskJson_dic = res_dic['taskJson']
        else:
            print("f5eCalc receive job error! errorCode: ",errorCode)
           
    except Exception as e:
        print("no f5eCalc job.")
       
    return taskJson_dic, errorCode, errorMsg
    
def fea_calc_output(input_dic,output_url,emb):
    #TODO
    r = []
    try:
      
        feature = array64_to_base64(np.array(emb))     
        #feature = emb
        #print(feature)
        sid = input_dic['sid']
        location = input_dic['location']
        container_id = input_dic['containerId']
        topNum = 5
        out_dic = {"feature":feature,'topNum':topNum,'location':location,
                   "camId":input_dic["camId"],"capTs":input_dic["capTs"],'sid':sid,'containerId':container_id}
        r = requests.post(output_url, data=out_dic)
        #print(r.content)
        #print(r.text)

    except Exception as e:
        print("f5eCalc post result error")
        print(r.text)
        
# =============================================================================
#     res = r.json()
#     errorCode = res['errorCode']
#     errorMsg = res['errorMsg']
#     print("error Code:",errorCode)
#     print("errMSg:", errorMsg)
# =============================================================================
    
def base64_to_image(base64_code):
    # 
    img_data = base64.b64decode(base64_code)
    # 
    img_array = np.frombuffer(img_data, np.uint8)
    # 
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)

    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def image_to_base64(image_np):

    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    image = cv2.imencode('.jpg',image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code    

def array64_to_base64(array):
    return base64.b64encode(array)

def base64_to_array64(base64_code):
    data = base64.b64decode(base64_code)
    array = np.frombuffer(data, np.float64)
    return array

def read_img_from_fea_taskJson(task_dic,tsb):
    storage = task_dic['storage']
    img  = []
    if storage == 1:
        img_path = task_dic['imagePath']    
        try:
            img = imageio.imread(img_path)
        except Exception as e:
            msg = 'Fea-Calc failed.'
           # write_logs(log_path,ai_type,task_dic,msg,tsb,time.time())
            img = []
            errorMessage = '{}: {}'.format(img_path, e)
            print(errorMessage)
            
    elif  storage == 3:
         try:
            print("start read base 64img")
            base64_code = task_dic['avatar'] 
            print("start read base_64_code.")
            img = base64_to_image(base64_code)   
            print("decode base_64img success.")      # 把二进制文件解码，并复制给data
         except Exception as e:
            msg = 'decode base_64img failed. '
            #write_logs(log_path,ai_type,task_dic,msg,tsb,time.time())
           
            print(msg)

    return img

def write_logs(log_path,ai_type,ai_uuid,ai_status,taskJson_dic,out_nums,tsb,tse):
    
    tsb = int(1000 * tsb)
    tse = int(1000 * tse)
    cam_id = '00'
    cap_ts = '00'
    if len(taskJson_dic)>0: 
        cam_id = taskJson_dic['camId']
        cap_ts = taskJson_dic['capTs']
        
    (filepath,tempfilename) = os.path.split(log_path)
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    with open(log_path,'at') as f:
        f.write('module:%s\tuuid:%s\tcapTs:%s\tcamId:%s\ttsb:%s\ttse:%s\tstatus:%d\toutputs:%d\n' % (ai_type, ai_uuid,cap_ts, cam_id,\
                                         tsb, tse,ai_status,out_nums))  

def cosine_similarity(embeddings1,embeddings2):
    if embeddings1.ndim == 1:
        embeddings1 = np.expand_dims(embeddings1,axis=0)
    if embeddings2.ndim == 1:
        embeddings2 = np.expand_dims(embeddings2,axis=0)
    # Distance based on cosine similarity
    dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1) 
    #dot = np.sum(embeddings1*embeddings2, axis = 1)
    norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
    similarity = dot / norm
    #dist = np.arccos(similarity) / math.pi
    return similarity    
    
 

def load_facenet_model(model_dir):
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            facenet.load_model(model_dir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
    return sess,images_placeholder,embeddings,phase_train_placeholder

def feature_extraction(sess,images_placeholder,embeddings,phase_train_placeholder,img_rgb,image_size):

    # Run forward pass to calculate embeddings
    #start_time = time.time()
      
    image = prewhiten_img_to_tensor(img_rgb,image_size)
    feed_dict = { images_placeholder: image, phase_train_placeholder:False }
    # Use the facenet model to calcualte embeddings
    embed = sess.run(embeddings, feed_dict=feed_dict)
    
    #run_time = time.time() - start_time
    #print('feature extraction run time: ', run_time)    
    
    return embed[0]  

def main(args):

# =============================================================================
#     img_path = "/home/luoyuhao/Datasets/Docker/1.png"
#     img_io = imageio.imread(img_path)
#  
#     base64_img = image_to_base64(img_io)
#     
#     sess,images_placeholder,embeddings,phase_train_placeholder = load_facenet_model(args.model_dir)
#     
#     emb_array = feature_extraction(sess,images_placeholder,embeddings,phase_train_placeholder,\
#                                    base64_img,args.image_size)
# 
# =============================================================================
   
        
    input_url,output_url,log_path,ai_uuid = read_config(config_path)
   
    #input_url = 'http://14.152.78.59:9090/f5eCalcQueue/popTask'
    #output_url = 'http://14.152.78.59:9090/f5eMatchQueue/pushTask'

    sess,images_placeholder,embeddings,phase_train_placeholder = load_facenet_model(args.model_dir)

    while(1):
        #time.sleep(1)
        tsb = time.time()
        ai_status = 1
        if read_state(job_path):
        #if True:    
            taskJson_dic, errorCode, errorMsg = read_input(input_url)
# =============================================================================
#             print("taskJson_dic:",taskJson_dic)
#             print("errorCode:",errorCode)
#             print("errorMsg",errorMsg)
# =============================================================================
            if len(taskJson_dic)!=7:
                continue
            
            img = read_img_from_fea_taskJson(taskJson_dic,tsb)
            
            if len(img)>0:
                try:
                    #print("start feacture Calc")
                    
                    emb_array = feature_extraction(sess,images_placeholder,embeddings,phase_train_placeholder,\
                                   img,args.image_size)
                    #print("Calc finish.")
                    print("feature dim:")
                    print(emb_array.shape)
                    fea_calc_output(taskJson_dic,output_url,emb_array)
                   
                    write_logs(log_path,ai_type,ai_uuid,ai_status,taskJson_dic,1,tsb,time.time())
                    print('f5eCalc success.')
                
                except Exception as e:
                    print("f5eCalc process failed. ") 
                    ai_status = -1
                    write_logs(log_path,ai_type,ai_uuid,ai_status,taskJson_dic,0,tsb,time.time())
        else:
             taskJson_dic = []
             ai_status = 2
             write_logs(log_path,ai_type,ai_uuid,ai_status,taskJson_dic,0,tsb,time.time())
             time.sleep(1)
             #msg = ai_type +' is sleep.' 
             #write_logs(log_path,ai_type,taskJson_dic,msg,tsb,time.time())

#######################################################################################################
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str,
        help='Directory containing the meta_file and ckpt_file')
  
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)

    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.',
        default=1.0)
    parser.add_argument('--image_batch', type=int,
        help='Number of images stored in memory at a time. Default 500.',
        default=500)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    #main(parse_arguments(sys.argv[1:]))
    #args = ['/root/facenet/src/20180402-114759']
    
    #args = ['/home/luoyuhao/facenet/src/20180402-114759']    512
    args  = ['/root/facenet/src/20180930-174542']    # 256
    
    main(parse_arguments(args))
