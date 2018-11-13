# -*- coding: utf-8 -*-
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

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import align.detect_face
import time
import imageio
import requests
import cv2
import base64
#import skimage
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

ai_type = 'faceDetect'
ai_uuid = ''
ai_status = -1
config_path = '/data/common/common.json'
job_path = '/data/job/job.json'
log_path = ''

test_config_path = '/home/luoyuhao/Datasets/Docker/configure.json'
test_job_path = '/home/luoyuhao/Datasets/Docker/job.json'


def load_mtcnn_model(args):
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            start_time = time.time();
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            print('create and load mtcnn model time: ', (time.time() - start_time))
    
    return pnet,rnet,onet
    
def crop_face(img,bounding_boxes,margin,image_size):
    nrof_faces = bounding_boxes.shape[0]
    face_res = []
    if nrof_faces>0:
        det = bounding_boxes[:,0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces>1:
            for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
        else:
            det_arr.append(np.squeeze(det))
            
        
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            #scaled = skimage.transform.resize(cropped, (args.image_size, args.image_size), interp='bilinear')
            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            face_res.append(scaled)
    return face_res

def save_faces(res,output_filename):
    if(len(res)>0):
        filename_base, file_extension = os.path.splitext(output_filename)
    for i in range(len(res)):
        if (len(res)>1):
            output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
        else:
            output_filename_n = "{}{}".format(filename_base, file_extension)
        imageio.imwrite(output_filename_n, res[i])

def img_resize(img,scale):
    
    img_resize = misc.imresize(img, (int(img.shape[0]/scale), int(img.shape[1]/scale)), interp='bilinear')
    return img_resize

def detectFace(args,img,pnet,rnet,onet,output_filename=None,isDrawFace=False,isPrintTimeInfo=False):
    ## reseize img to detect
    if args.scale>1:
        img_input = img_resize(img,args.scale)
    else:
        img_input = img
    

    detect_time_start = time.time()
    bounding_boxes, _ = align.detect_face.detect_face(img_input, minsize, pnet, rnet, onet, threshold, factor)
    if args.scale>1:
        bounding_boxes[:,0:4] = args.scale * bounding_boxes[:,0:4]
        
    detect_time = time.time() - detect_time_start
    if isPrintTimeInfo:
        print('detect_face_time: ', detect_time)  
        
    faces = crop_face(img,bounding_boxes,args.margin,args.image_size)
    
    if(output_filename is not None):
        save_time_start = time.time()
        save_faces(faces,output_filename)
        if isPrintTimeInfo:
            print('save_face_time: ', time.time() - save_time_start) 

    
    if isDrawFace:
        draw = align.detect_face.drawBoxes(img,bounding_boxes)
        filename_base, file_extension = os.path.splitext(output_filename)
        imageio.imwrite(filename_base+'_res'+file_extension,draw)
    #print(bounding_boxes)
    return faces,bounding_boxes

    
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
        jobpath = dic['job']
        #jobpath = '/home/luoyuhao/Datasets/Docker/job.json'
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
    r = []
    taskJson_dic = []
    errorCode = []
    errorMsg = []   
    try:
        r = requests.get(input_url)
        res_dic = r.json()   #dic
    except Exception as e:
        #errorMessage = '{}: {}'.format(input_url, e)
        print("faceDetect requests get error.")
    
    try:
        errorCode = res_dic['errorCode']
        errorMsg = res_dic['errorMsg']
        if errorCode == 0:
             taskJson_dic = res_dic['taskJson']
        else:
            print("faceDetect receive job error! errorCode: ",errorCode)
            
    except Exception as e:
        print("no face detect job.")
   
    return taskJson_dic, errorCode, errorMsg
   
def push_output(input_dic,output_url,faces,bounding_boxes):
    #TODO
    r = []
    try:
        for i in range(len(faces)):
            storage = input_dic['storage']
            if storage == 1:
                save_path = '/home/luoyuhao/Datasets/Docker/saveface/'
                save_path = save_path + str(time.time())+".png"
                #imageio.imwrite(save_path,faces[i])
                avatar = save_path
    
            elif storage == 3:
                avatar = image_to_base64(np.array(faces[i]))
                
    
            box = bounding_boxes[i,0:4]
            location = [int(box[0]),int(box[1]),int(box[2]),int(box[3])]
            out_dic = {"storage":storage,"avatar":avatar,'location':str(location),
                       "camId":input_dic["camId"],"capTs":input_dic["capTs"],
                       'sid':input_dic['sid'],'containerId':input_dic['containerId']}
                
            r = requests.post(output_url, data=out_dic)
    except Exception as e:
        print("faceDetect post result error.")
        print(r.text)
    
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

def read_img_from_taskJson(task_dic,tsb):
    storage = task_dic['storage']
    img  = []
    if storage == 1:
        img_path = task_dic['imagePath']    
        try:
            img = imageio.imread(img_path)
        except Exception as e:
            msg = 'Face-detect failed. Wrong picture format'
            #write_logs(log_path,ai_type,task_dic,msg,tsb,time.time())
            img = []
            errorMessage = '{}: {}'.format(img_path, e)
            print(errorMessage)
            
    elif  storage == 3:
         try:
            base64_code = task_dic['imagePath'] 
            img = base64_to_image(base64_code)      # 把二进制文件解码，并复制给data

         except Exception as e:
            msg = 'Face-detect failed. Wrong picture format'
            img = []


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

# =============================================================================
# 
# def write_logs(log_path,ai_type,taskJson_dic,msg,tsb,tse):
#     
#     tsb = int(1000 * tsb)
#     tse = int(1000* tse)
#     if len(taskJson_dic)>0: 
#         storage = taskJson_dic['storage']
#         img_path = taskJson_dic['imagePath']
#         cam_id = taskJson_dic['camId']
#         cap_ts = taskJson_dic['capTs']
#     else:
#         storage = ''
#         img_path = ''
#         cam_id = ''
#         cap_ts = ''
#         
#     (filepath,tempfilename) = os.path.split(log_path)
#     if not os.path.exists(filepath):
#         os.mkdir(filepath)
#     with open(log_path,'at') as f:
#         f.write('tsb:%s\ttype:%s\tstorage:%d\timagePath:%s\tcamId:%d\tcapTs:%d\tmsg:%s\ttse:%s\n' % (tsb, ai_type,storage, img_path,\
#                                          cam_id, cap_ts,msg,tse))
# =============================================================================
        
########################################################################################################
def main(args):

# =============================================================================
#     output_dir = args.output_dir
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)          
#     #
#     image_path = args.image_path
#     img = imageio.imread(image_path)
#     filename = os.path.splitext(os.path.split(image_path)[1])[0]
#     output_filename = os.path.join(output_dir, filename+'.png')
#     
# =============================================================================
    #input_url,output_url,log_path,ai_uuid = read_config(config_path)
    input_url = 'http://14.152.78.59:9090/faceDetectQueue/popTask'
    output_url = 'http://14.152.78.59:9090/f5eCalcQueue/pushTask'
    
    pnet,rnet,onet = load_mtcnn_model(args)
    
    while(1):
        #time.sleep(2)
        tsb = time.time()
        ai_status = 1
        #if read_state(job_path):
        if True:   
            taskJson_dic, errorCode, errorMsg = read_input(input_url)
            #print(taskJson_dic)
            if len(taskJson_dic)!=6:
                continue
            
            img = read_img_from_taskJson(taskJson_dic,tsb)
            
            if len(img)>0:
                try:
                    faces, bounding_boxes = detectFace(args,img,pnet,rnet,onet,None,False,True)
                    push_output(taskJson_dic,output_url,faces,bounding_boxes)
                    nums = len(faces)
                    #write_logs(log_path,ai_type,ai_uuid,ai_status,taskJson_dic,nums,tsb,time.time())
                    msg = 'Face-detect success. find {} faces'.format(nums)
                    print(msg)
                
                except Exception as e:
                    print("Face detect process failed.") 
                    ai_status = -1
                   # write_logs(log_path,ai_type,taskJson_dic,msg,tsb,time.time())
        else:
             taskJson_dic = []
             ai_status = 2
             #write_logs(log_path,ai_type,ai_uuid,ai_status,taskJson_dic,0,tsb,time.time())
             time.sleep(1)
             #msg = ai_type +' is sleep.' 
             #write_logs(log_path,ai_type,taskJson_dic,msg,tsb,time.time())
             
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)
    parser.add_argument('--scale', type=int,
                        help='the height and width will resize to height/scale and width/scale to detect faces.', default=2)
    return parser.parse_args(argv)

if __name__ == '__main__':
    
    args = ['--scale','3']
    main(parse_arguments(args))
    #main(parse_arguments(sys.argv[1:]))
   
