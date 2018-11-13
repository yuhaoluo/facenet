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
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

ai_type = 'f5eCalc'
ai_uuid = ''
ai_status = -1
config_path = '/data/common/common.json'
job_path = '/data/job/job.json'
log_path = ''

####################################################################################################

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


############################################################################################
# read common.json
def read_config(common_json_path,job_json_path,fail_log_path,ai_type):
    #TODO
    dic={}
    try:
        f = open(common_json_path)     
        json_read = f.read()
        dic = json.loads(json_read)
        input_url = dic["input"]
        output_url = dic['output']
        #job_json_path = dic['job']
        f.close()
        uuid,state = read_job(job_json_path,fail_log_path,ai_type)
        logs_path = '/data/logs/'+str(uuid)+'.logs'
        return input_url,output_url,logs_path,uuid
    except Exception as e:
        #print(e)
        write_msg(fail_log_path,ai_type,'before get uuid',e,Times(time),camId='',capTs='')
        msg = 'read ' + common_json_path + ' fail.'
        write_msg(fail_log_path,ai_type,'before get uuid',msg,Times(time),camId='',capTs='')
        return 
      
# read job.json    
def read_job(job_path,fail_log_path,ai_type):
    try:
        f = open(job_path)     
        json_read = f.read()
        dic = json.loads(json_read)
        uuid = dic['Uuid']
        state = dic['run']
        f.close()
        return uuid,state
    except Exception as e:
        #print(e)
        write_msg(fail_log_path,ai_type,'before get uuid',e,Times(time),camId='',capTs='')
        msg = 'read ' + job_path + ' fail.'
        write_msg(fail_log_path,ai_type,'before get uuid',msg,Times(time),camId='',capTs='')
        return 
    

def read_state(job_path,log_path,ai_type,ai_uuid):
    #TODO
    try:
        if not os.path.exists(job_path):
            msg = job_path+ ' is not exist.'
            write_msg(log_path,ai_type,ai_uuid,msg,Times(time),camId='',capTs='')
        else:
            f = open(job_path)     
            json_read = f.read()
            dic = json.loads(json_read)
            state = dic['run']
            f.close()
    except Exception as e:
        print(e)
        write_msg(log_path,ai_type,ai_uuid,e,Times(time),camId='',capTs='')
        msg = 'read '+ job_path + " fail."
        write_msg(log_path,ai_type,ai_uuid,msg,Times(time),camId='',capTs='')

    if state == 'true':
        return True
    else:
        return False

def read_input(input_url,log_path,ai_type,ai_uuid):
    #TODO
    r = []
    res_dic = []
    taskJson_dic = []
    errorCode = []
    errorMsg = []   
    try:
        r = requests.get(input_url)
        if r.raise_for_status() is None:
            res_dic = r.json()   #dic
        else:
            msg = input_url+ "respose status_code is :" + str(r.status_code)
            write_msg(log_path,ai_type,ai_uuid,msg,Times(time),camId='',capTs='') 
        
    except Exception as e:
        #msg = input_url+ "respose status_code is :" + str(r.status_code)
        write_msg(log_path,ai_type,ai_uuid,e,Times(time),camId='',capTs='') 

        print("faceCalc requests.get {} error".format(input_url))
    
    try:
        errorCode = res_dic['errorCode']
        errorMsg = res_dic['errorMsg']
        if errorCode == 0:
             taskJson_dic = res_dic['taskJson']
        else:
            
            print("faceCalc requests.get json errorCode: ",errorCode)
            msg = "faceCalc requests.get {} json errorCode: {}".format(input_url,errorCode)
            write_msg(log_path,ai_type,ai_uuid,msg,Times(time),camId='',capTs='') 
            
    except Exception as e:
        msg = "no faceCalc job. 'taskJson' is not exist"
        print(msg)
        write_msg(log_path,ai_type,ai_uuid,msg,Times(time),camId='',capTs='')
   
    return taskJson_dic, errorCode, errorMsg
###############################################


    
def fea_calc_output(input_dic,output_url,emb,log_path,ai_type,ai_uuid):
    #TODO
    r = []
    try:
      
        feature = array64_to_base64(np.array(emb))    
        msg = 'feature (dtype:{}, dim:{})  to base64 success'.format(emb.dtype,emb.shape)
        print(msg)
        write_msg(log_path,ai_type,ai_uuid,msg,Times(time),input_dic["camId"],input_dic["capTs"])
        #feature = emb
        #print(feature)
        sid = input_dic['sid']
        location = input_dic['location']
        container_id = input_dic['containerId']
        topNum = 5
        out_dic = {"feature":feature,'topNum':topNum,'location':location,
                   "camId":input_dic["camId"],"capTs":input_dic["capTs"],'sid':sid,'containerId':container_id}
        #r = requests.post(output_url, data=out_dic)
        
        try:
            r = requests.post(output_url, data=out_dic)
            
            if r.raise_for_status() is not None:
                msg = 'faceCalc requests.post {} respose status code is:{}'.format(output_url,str(r.status_code))
                print(msg)
                write_msg(log_path,ai_type,ai_uuid,msg,Times(time),input_dic["camId"],input_dic["capTs"])
                #print(r.text)
        except Exception as e:
            write_msg(log_path,ai_type,ai_uuid,e,Times(time),input_dic["camId"],input_dic["capTs"])
        #print(r.content)
        #print(r.text)

    except Exception as e:
        write_msg(log_path,ai_type,ai_uuid,e,Times(time),input_dic["camId"],input_dic["capTs"])
        print("faceCalc post result error.")
        #print(r.text)
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

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

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

def read_img_from_fea_taskJson(task_dic,log_path,ai_type,ai_uuid):
    storage = task_dic['storage']
    img  = []
    cam_id = []
    cap_ts = []    
    if storage == 1:
        img_path = task_dic['imagePath']    
        try:
            cam_id = task_dic['camId']
            cap_ts = task_dic['capTs']
            img = imageio.imread(img_path)
        except Exception as e:
            msg = 'read image: {} fail.'.format(img_path)
            write_msg(log_path,ai_type,ai_uuid,e,Times(time),cam_id,cap_ts) 
            
    elif  storage == 3:
         try:
            cam_id = task_dic['camId']
            cap_ts = task_dic['capTs']
            #print("start read base 64img")
            base64_code = task_dic['avatar'] 
            #print("start read base_64_code.")
            img = base64_to_image(base64_code)   
            msg = "decode base_64img success."      # 把二进制文件解码，并复制给data
            write_msg(log_path,ai_type,ai_uuid,msg,Times(time),cam_id,cap_ts) 
         except Exception as e:
            msg = 'decode base_64img failed. '
            print(msg)
            write_msg(log_path,ai_type,ai_uuid,e,Times(time),cam_id,cap_ts)
            write_msg(log_path,ai_type,ai_uuid,msg,Times(time),cam_id,cap_ts)
          
    return img

def write_logs(log_path,ai_type,ai_uuid,ai_status,taskJson_dic,out_nums,tsb,tse,date_time):
    
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
        f.write('date:%s\tmodule:%s\tuuid:%s\tcapTs:%s\tcamId:%s\ttsb:%s\ttse:%s\tstatus:%d\toutputs:%d\n' % (date_time,ai_type, ai_uuid,cap_ts, cam_id,\
                                         tsb, tse,ai_status,out_nums))  

def Times(time):
    date_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    return date_time

def write_msg(log_path,ai_type,ai_uuid,msg,date_time,camId='',capTs=''):
    (filepath,tempfilename) = os.path.split(log_path)
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    with open(log_path,'at') as f:
        f.write('date:%s\tmodule:%s\tuuid:%s\tmsg:%s\tcamId:%s\tcapTs:%s\n' % (date_time,ai_type,ai_uuid,msg,camId,capTs)) 


def checkStart(config_path,job_path,fail_log_path):
    start = True
    while(start):
        #fail_log_path = '/data/logs/fail.logs'
        if not os.path.exists(config_path):
            start = True
            msg = config_path +" is not exist."
            
            (filepath,tempfilename) = os.path.split(fail_log_path)
            if not os.path.exists(filepath):
                os.mkdir(filepath) 
            write_msg(fail_log_path,ai_type,'before get uuid',msg,Times(time))
        else:
            start = False
    
        if not os.path.exists(job_path):
            start = True
            msg = job_path +" is not exist."
            #fail_log_path = '/data/logs/fail.logs'
            (filepath,tempfilename) = os.path.split(fail_log_path)
            if not os.path.exists(filepath):
                os.mkdir(filepath) 
            write_msg(fail_log_path,ai_type,'before get uuid',msg,Times(time))      
        else:
            start = False
        if(start):
            time.sleep(5)

###################################################################################################
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
##################################################################################################
def main(args):

    fail_log_path = '/data/logs/fail.logs' 
    checkStart(config_path,job_path,fail_log_path)
            
    input_url,output_url,log_path,ai_uuid = read_config(config_path,job_path,fail_log_path,ai_type)
    
    write_msg(log_path,ai_type,ai_uuid,input_url,Times(time))
    write_msg(log_path,ai_type,ai_uuid,output_url,Times(time))
    write_msg(log_path,ai_type,ai_uuid,log_path,Times(time))

    #input_url = 'http://14.152.78.59:9090/f5eCalcQueue/popTask'
    #output_url = 'http://14.152.78.59:9090/f5eMatchQueue/pushTask'
    
    try:
        
        sess,images_placeholder,embeddings,phase_train_placeholder = load_facenet_model(args.model_dir)
        msg = 'create facenet model done.'
        write_msg(log_path,ai_type,ai_uuid,msg,Times(time))
    except Exception as e:
        msg = 'create facenet model fail.'
        write_msg(log_path,ai_type,ai_uuid,msg,Times(time))

    while(1):
        #time.sleep(1)
        tsb = time.time()
        ai_status = 1
        if read_state(job_path,log_path,ai_type,ai_uuid):
        #if True:    
            taskJson_dic, errorCode, errorMsg = read_input(input_url,log_path,ai_type,ai_uuid)

            if len(taskJson_dic)!=7:
                continue
            
            img = read_img_from_fea_taskJson(taskJson_dic,log_path,ai_type,ai_uuid)
            
            if len(img)>0:
                try:
                    #print("start feacture Calc")
                    cam_id = taskJson_dic['camId']
                    cap_ts = taskJson_dic['capTs']
                    emb_array = feature_extraction(sess,images_placeholder,embeddings,phase_train_placeholder,\
                                   img,args.image_size)
                    msg = "f5eCalc calc feataure complete."
                    print(msg)
                    write_msg(log_path,ai_type,ai_uuid,msg,Times(time),cam_id,cap_ts)
                    #print("feature dim:")
                    #print(emb_array.shape)
                    fea_calc_output(taskJson_dic,output_url,emb_array,log_path,ai_type,ai_uuid)
                    msg = 'f5eCalc output to {} success.'.format(output_url)
                    print(msg)
                    write_logs(log_path,ai_type,ai_uuid,ai_status,taskJson_dic,1,tsb,time.time(),Times(time))
                    
                
                except Exception as e:
                    print("f5eCalc process failed. error msg: {} ".format(e)) 
                    write_msg(log_path,ai_type,ai_uuid,e,Times(time),cam_id,cap_ts)
                   
                    ai_status = -1
                    taskJson_dic = []
                    write_logs(log_path,ai_type,ai_uuid,ai_status,taskJson_dic,0,tsb,time.time(),Times(time))
        else:
             taskJson_dic = []
             ai_status = 2
             write_logs(log_path,ai_type,ai_uuid,ai_status,taskJson_dic,0,tsb,time.time(),Times(time))
             time.sleep(1)
             write_msg(log_path,ai_type,ai_uuid,'wait',Times(time))
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
