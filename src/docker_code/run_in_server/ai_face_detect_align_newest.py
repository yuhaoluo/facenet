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
import sys
import codecs
import face_preprocess
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

minsize = 20 # minimum size of face
threshold = [ 0.99, 0.99, 0.99 ]  # three steps's threshold
factor = 0.709 # scale factor

ai_type = 'faceDetect'
ai_uuid = ''
ai_status = -1
config_path = '/data/common/common.json'
job_path = '/data/job/job.json'
log_path = ''
#fail_log_path = '/data/logs/fail.logs'

test_config_path = '/home/luoyuhao/Datasets/Docker/configure.json'
test_job_path = '/home/luoyuhao/Datasets/Docker/job.json'

########################################################################################################

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

def crop_face_with_align(img,bounding_boxes,landmark,margin,image_size):
    nrof_faces = bounding_boxes.shape[0]
    face_res = []
    if nrof_faces>0:
        det = bounding_boxes[:,0:4]
        det_arr = []
        if nrof_faces>1:
            for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
        else:
            det_arr.append(np.squeeze(det))
            
        for i, det in enumerate(det_arr):
            point = landmark[:,i]
            
            point_x = point[0:5]
            point_y = point[5:10]
            mark = np.zeros((5,2))
            mark[:,0] = point_x
            mark[:,1] = point_y
            #print(arr2)      
            cropped = face_preprocess.preprocess(img, det, mark, image_size='112,112')                        
            face_res.append(cropped)
    return face_res

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
            
        face_res = []
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
    bounding_boxes, points = align.detect_face.detect_face(img_input, minsize, pnet, rnet, onet, threshold, factor)
    if args.scale>1:
        bounding_boxes[:,0:4] = args.scale * bounding_boxes[:,0:4]
        points = args.scale * points
    detect_time = time.time() - detect_time_start
    if isPrintTimeInfo:
        print('detect_face_time: ', detect_time)  
        
    faces = crop_face_with_align(img,bounding_boxes,args.margin,args.image_size)
    
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

        print("faceDetect requests.get {} error".format(input_url))
    
    try:
        errorCode = res_dic['errorCode']
        errorMsg = res_dic['errorMsg']
        if errorCode == 0:
             taskJson_dic = res_dic['taskJson']
        else:
            
            print("faceDetect requests.get json errorCode: ",errorCode)
            msg = "faceDetect requests.get {} json errorCode: {}".format(input_url,errorCode)
            write_msg(log_path,ai_type,ai_uuid,msg,Times(time),camId='',capTs='') 
            
    except Exception as e:
        msg = "no face detect job. 'taskJson' is not exist"
        print(msg)
        write_msg(log_path,ai_type,ai_uuid,msg,Times(time),camId='',capTs='')
   
    return taskJson_dic, errorCode, errorMsg
   
def push_output(input_dic,output_url,faces,bounding_boxes,log_path,ai_type,ai_uuid):
    #TODO
    r = []
    try:
        for i in range(len(faces)):
            storage = input_dic['storage']
            if storage == 1:
                #save_path = '/home/luoyuhao/Datasets/Docker/saveface/'
                #save_path = save_path + str(time.time())+".png"
                #imageio.imwrite(save_path,faces[i])
                avatar = image_to_base64(np.array(faces[i]))
                storage = 3
    
            elif storage == 3:
                avatar = image_to_base64(np.array(faces[i]))
                
            box = bounding_boxes[i,0:4]
            location = [int(box[0]),int(box[1]),int(box[2]),int(box[3])]
            out_dic = {"storage":storage,"avatar":avatar,'location':str(location),
                       "camId":input_dic["camId"],"capTs":input_dic["capTs"],
                       'sid':input_dic['sid'],'containerId':input_dic['containerId']}
            
            try:
                r = requests.post(output_url, data=out_dic)
                
                if r.raise_for_status() is not None:
                    msg = 'facedetect requests.post {} respose status code is:{}'.format(output_url,str(r.status_code))
                    write_msg(log_path,ai_type,ai_uuid,msg,Times(time),input_dic["camId"],input_dic["capTs"])
                    print(msg)
            except Exception as e:
                write_msg(log_path,ai_type,ai_uuid,e,Times(time),input_dic["camId"],input_dic["capTs"])
                
    except Exception as e:
        write_msg(log_path,ai_type,ai_uuid,e,Times(time),input_dic["camId"],input_dic["capTs"])
        print("faceDetect post result error.")
        #print(r.text)
    
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

def read_img_from_taskJson(task_dic,log_path,ai_type,ai_uuid):
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
            write_msg(log_path,ai_type,ai_uuid,e,Times(time),task_dic['camId'],task_dic['capTs']) 
            
    elif  storage == 3:
         try:

            base64_code = task_dic['imagePath']
            cam_id = task_dic['camId']
            cap_ts = task_dic['capTs']
            img = base64_to_image(base64_code)      # 把二进制文件解码，并复制给data
            #cam_id = task_dic['camId']
            #cap_ts = task_dic['capTs']
         except Exception as e:
             write_msg(log_path,ai_type,ai_uuid,e,Times(time),cam_id,cap_ts) 
             msg = 'decode base64 fail'
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

def save_faces_to_docker2(res,docker_folder):
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


def save_faces_to_docker(res,docker_folder,cam_id,cap_ts,index=1):
    if not os.path.exists(docker_folder):
        os.makedirs(docker_folder)
    for i in range(len(res)):

        filename = str(cam_id)+'-'+str(cap_ts)+'-'+str(index)+'.jpg'
        output_filename = os.path.join(docker_folder,filename)
        imageio.imwrite(output_filename,res[i])
        index = index+1
    return index  
########################################################################################################
def main(args):

    #fail_log_path = '/home/luoyuhao/Test/fail.logs' 
    docker_folder = '/data/logs/images/'
    fail_log_path = '/data/logs/fail.logs' 
    checkStart(config_path,job_path,fail_log_path)
            
    input_url,output_url,log_path,ai_uuid = read_config(config_path,job_path,fail_log_path,ai_type)
    
    write_msg(log_path,ai_type,ai_uuid,input_url,Times(time))
    write_msg(log_path,ai_type,ai_uuid,output_url,Times(time))
    write_msg(log_path,ai_type,ai_uuid,log_path,Times(time))
    
    #input_url = 'http://14.152.78.59:9090/faceDetectQueue/popTask'
    #output_url = 'http://14.152.78.59:9090/f5eCalcQueue/pushTask'
    try:
        pnet,rnet,onet = load_mtcnn_model(args)
        msg = 'create mtcnn model done.'
        write_msg(log_path,ai_type,ai_uuid,msg,Times(time))
    except Exception as e:
        msg = 'create mtcnn model fail.'
        write_msg(log_path,ai_type,ai_uuid,msg,Times(time))
    index = 1   
    while(1):
        #time.sleep(2)
        tsb = time.time()
        ai_status = 1
        if read_state(job_path,log_path,ai_type,ai_uuid):
        #if True:   
            taskJson_dic, errorCode, errorMsg = read_input(input_url,log_path,ai_type,ai_uuid)
            #print(taskJson_dic)
            if len(taskJson_dic)!=6:
                msg = 'json miss some params.'
                write_msg(log_path,ai_type,ai_uuid,msg,Times(time))
                continue
            
            img = read_img_from_taskJson(taskJson_dic,log_path,ai_type,ai_uuid)
            cam_id = []
            cap_ts = []
            if len(img)>0:
                try:
                    cam_id = taskJson_dic['camId']
                    cap_ts = taskJson_dic['capTs']
                    faces, bounding_boxes = detectFace(args,img,pnet,rnet,onet,None,False,True)
                    push_output(taskJson_dic,output_url,faces,bounding_boxes,log_path,ai_type,ai_uuid)
                    nums = len(faces)
                    msg = 'Face-detect success. find {} faces'.format(nums)
                    print(msg)
                    write_logs(log_path,ai_type,ai_uuid,ai_status,taskJson_dic,nums,tsb,time.time(),Times(time))
                    
                    if nums>0:
                        index = save_faces_to_docker(faces,docker_folder,cam_id,cap_ts,index)
                        msg = "Save faces {} success".format(nums)
                        write_msg(log_path,ai_type,ai_uuid,msg,Times(time))
                        print(msg)
                except Exception as e:
                    print(e) 
                    write_msg(log_path,ai_type,ai_uuid,e,Times(time),cam_id,cap_ts)
                    ai_status = -1
                    taskJson_dic = []
                    write_logs(log_path,ai_type,ai_uuid,ai_status,taskJson_dic,0,tsb,time.time(),Times(time))
        else:
             taskJson_dic = []
             ai_status = 2
             write_logs(log_path,ai_type,ai_uuid,ai_status,taskJson_dic,0,tsb,time.time(),Times(time))
             write_msg(log_path,ai_type,ai_uuid,'wait',Times(time))
             time.sleep(1)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=32)
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
   
