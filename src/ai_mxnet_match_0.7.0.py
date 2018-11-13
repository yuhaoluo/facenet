# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:27:42 2018

@author: luoyuhao
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import os
import base64
import json
import requests
import heapq
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
#import imp
#imp.reload(sys)
#sys.setdefaultencoding('utf8')

ai_type = 'f5eMatch'
ai_uuid = ''
ai_status = -1
config_path = '/data/common/common.json'
job_path = '/data/job/job.json'
log_path = ''
old_db_list =[]
face_lib_list = []
# read common.json
def read_config(config_path):
    #TODO
    dic={}
    try:
        f = open(config_path)     
        json_read = f.read()
        dic = json.loads(json_read,encoding="utf-8")
        input_url = dic["input"]
        output_url = dic['output']
        jobpath = dic['job']
        #jobpath = '/home/luoyuhao/Datasets/Docker/job.json'
        f.close()
        uuid,state = read_job(jobpath)
        logs_path = '/data/logs/'+str(uuid)+'.logs'
        return input_url,output_url,logs_path,uuid
    except Exception as e:
        print("read config json fail.")
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
    state = []
    try:
        f = open(job_path)     
        json_read = f.read()
        dic = json.loads(json_read,encoding="utf-8")
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
    taskJson_dic = []
    errorCode = []
    errorMsg = []   
    r = []

    try:
        r = requests.get(input_url)
        res_dic = r.json()   #dic
    except Exception as e:
        #errorMessage = '{}: {}'.format(input_url, e)
        print("f5eMatch request get error.")
    
    try:
        errorCode = res_dic['errorCode']
        errorMsg = res_dic['errorMsg']
        if errorCode == 0:
             taskJson_dic = res_dic['taskJson']
        else:
            print(" f5eMatch job errorCode: ",errorCode)
    except Exception as e:
        print("no f5eMatch job.")

    return taskJson_dic, errorCode, errorMsg
   

    
def array64_to_base64(array):
    return base64.b64encode(array)

def base64_to_array64(base64_code):
    data = base64.b64decode(base64_code)
    array = np.frombuffer(data, np.float32)
    return array

def read_fea_from_taskJson(task_dic,tsb):
     db_list = []
     
     try:
        base64_code = task_dic['feature'] 
        fea = base64_to_array64(base64_code)      # 把二进制文件解码，并复制给data
        
        #fea = task_dic['feature']
        db_list = task_dic['f5eDB']
        return fea,db_list
     except Exception as e:
        msg = 'f5eMatch failed. read feature and db fail.'
        print(msg)
        return  

def read_db_params(db_list):
     db_info_list =[]
     try:
        for i in range(len(db_list)):
            db_type = db_list[i]['type']
            db_num = db_list[i]['num']
            db_top = db_list[i]['top']
            db_name = db_list[i]['name']
            dic = {'type':db_type,'num':db_num,'top':db_top,'name':db_name}
            db_info_list.append(dic)
     except Exception as e:
        print('f5eMatch failed. read_db_params fail.')
     return db_info_list
 
def db_cmp(new_db_list,old_db_list):  
    db_to_be_read =[]
    if len(old_db_list)>0:
        for i in range(len(new_db_list)):
            for j in range(len(old_db_list)):
                if new_db_list[i]['type']!=old_db_list[j]['type'] \
                and new_db_list[i]['num']!=old_db_list[j]['num'] \
                and  j == len(old_db_list) -1 :
                    db_to_be_read.append(new_db_list[i])
       
        for i in range(len(db_to_be_read)):
            old_db_list.append(db_to_be_read[i])
    elif len(old_db_list) == 0:
        old_db_list = new_db_list
        db_to_be_read = new_db_list
    return db_to_be_read,old_db_list

def get_face_db(db_info_list):
    #TODO
    #db_url = 'http://172.16.0.98:8085/'
    db_url = 'http://14.152.78.59:9090/api/getFeatureContent'
    label_feas = []
    #start = time.time()
    r = []
    try:
        
        cur_db = db_info_list
        out_dic = {}
        out_dic['type'] = cur_db['type']
        out_dic['number'] =  cur_db['num']
      
        r = requests.post(db_url, data=out_dic)
        #print(r.text)
        json_dic = r.json()
        errorCode = json_dic['errorCode']
        #errorMsg = json_dic['errorMsg']
        if errorCode == 0:
            label_feas = json_dic['labelfeatures']
            print("get db success!")

            #db_list.append(cur_label_feas)
        else:
            print(" get db fail,errorCode:",errorCode)
            print(r.text)
    except Exception as e: 
        print('get db fail in exception')
        print(r.text)
    #print("read db time:",time.time()-start)
    #print("get db success!")
    return label_feas 

def write_msg(log_path,ai_type,ai_uuid,msg):
    (filepath,tempfilename) = os.path.split(log_path)
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    with open(log_path,'at') as f:
        f.write('module:%s\tuuid:%s\tmsg:%s\n' % (ai_type, ai_uuid,msg))  

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

class FeaCmpRes(object):
    def __init__(self,label,similarity):
        self.label = label
        self.similarity = similarity
    # choose one from the follow
    def __lt__(self,other):#operator < 
        return self.similarity < other.similarity
    def __str__(self):
        return '(' + str(self.label)+',\'' + self.similarity + '\')'

def get_topK(fea,labels_feas,k):
    heap = []
    for i in range(len(labels_feas)):
        sim = cosine_similarity(fea,np.array(labels_feas[i]['feature']))
        if i<k:
            item = FeaCmpRes(labels_feas[i]['label'],sim)
            heapq.heappush(heap, item)
        else:
            min_sim = heap[0].similarity
            if sim > min_sim:
                heapq.heappop(heap)
                item = FeaCmpRes(labels_feas[i]['label'],sim)
                heapq.heappush(heap,item)
    return heapq.nlargest(k,heap)

def FeaCmpResToList(cmp_res):
    matchRes =[]
    for i in range(len(cmp_res)):
        #labels.append(cmp_res[i].label)
        #sim.append(cmp_res[i].similarity)
        item ={'labels':cmp_res[i].label,'similarity':cmp_res[i].similarity}
        matchRes.append(item)
    return matchRes


# =============================================================================
# [  {'res': 
#  [ {'labels': '149-刘行知', 'similarity': array([0.44504654])}, 
# {'labels' : '70-李岩', 'similarity': array([0.34678088])}, 
# {'labels': '166-伍江鹏', 'similarity': array([0.31223487])},
# {'labels': '181-罗彬', 'similarity': array([0.30727538])},
# {'labels': '95-冯圆', 'similarity': array([0.29601155])}], 
#  'param': 
#  {'type': 1, 'num': 1, 'top': 5, 'name': '测试'} } ]
# =============================================================================

def fea_match_output(input_dic,output_url,match_res,log_path):
    #TODO 
    r = []
    #msg = 'into fea_match_output'
    #write_msg(log_path,ai_type,ai_uuid,msg)
    try:
        for i in range(len(match_res)):
            param = match_res[i]['param']
            f5eType = param['type']
            f5eNum = param['num']
            f5eName = param['name']
            
            location = input_dic['location']
            #print("location:",location)
            #print("type:",type(location))
            
            loc = location.split(',')
            x0 = int(loc[0].split('[')[1])
            y0 = int(loc[1])
            x1 = int(loc[2])
            y1 = int(loc[3].split(']')[0])
            cur_res = match_res[i]['res']
            if(len(cur_res))<1:
                print("no match res")
                write_msg(log_path,ai_type,ai_uuid,'no match res')
            out = []
            for j in range(len(cur_res)):
                cur_label = cur_res[j]['labels']
                cur_similarity = cur_res[j]['similarity'][0]
                
            
                out_dic = {'f5eType':f5eType,'f5eNum':f5eNum,'f5eName':f5eName,
                           'label':cur_label,'assemblity':cur_similarity,
                           'x0':x0,'y0':y0,'x1':x1,'y1':y1,
                           "camId":input_dic["camId"],"capTs":input_dic["capTs"],
                           'sid':input_dic['sid'],'containerId':input_dic['containerId']}
                out.append(out_dic)
            #print(out)
            out_str = json.dumps(out,ensure_ascii=False)
            #print(out_str)
            #msg = 'before requests post'
            #write_msg(log_path,ai_type,ai_uuid,msg)
            header = {'Content-Type': 'application/json'}
            r = requests.post(output_url, data=out_str.encode('utf-8'),headers = header)
            #print(r.text)
            #msg = "f5eMatch requests post msg: " + r.text
            #write_msg(log_path,ai_type,ai_uuid,msg)
            #msg = str(out_dic)
            #write_msg(log_path,ai_type,ai_uuid,msg)
    except Exception as e:
        print(e)
        #print("f5eMatch post result error")
        #print(r.text)
        #print(r.status_code)
        msg = 'f5eMatch requeests post error'
        write_msg(log_path,ai_type,ai_uuid,msg)

def f5ematch_output(input_dic,output_url,db_params,topk_res,log_path):
    #TODO 
    r = []
    #msg = 'into fea_match_output'
    #write_msg(log_path,ai_type,ai_uuid,msg)
    if(len(topk_res))<1:
        print("no match res")
        write_msg(log_path,ai_type,ai_uuid,'no match res')
        return
    try:
       
        f5eType = db_params['type']
        f5eNum = db_params['num']
        f5eName = db_params['name']
        location = input_dic['location']
        #print("location:",location)
        #print("type:",type(location))
        loc = location.split(',')
        x0 = int(loc[0].split('[')[1])
        y0 = int(loc[1])
        x1 = int(loc[2])
        y1 = int(loc[3].split(']')[0])
         

        out = []
        for j in range(len(topk_res)):
            cur_label = topk_res[j]['labels']
            cur_similarity = topk_res[j]['similarity']
            out_dic = {'f5eType':f5eType,'f5eNum':f5eNum,'f5eName':f5eName,
                       'label':cur_label,'assemblity':cur_similarity,
                       'x0':x0,'y0':y0,'x1':x1,'y1':y1,
                       "camId":input_dic["camId"],"capTs":input_dic["capTs"],
                       'sid':input_dic['sid'],'containerId':input_dic['containerId']}
            out.append(out_dic)
        print(out)
        out_str = json.dumps(out,ensure_ascii=False)
        #print(out_str)
        #msg = 'before requests post'
        #write_msg(log_path,ai_type,ai_uuid,msg)
        header = {'Content-Type': 'application/json'}
        r = requests.post(output_url, data=out_str.encode('utf-8'),headers = header)
        print(r.text)
        #msg = "f5eMatch requests post msg: " + r.text
        #write_msg(log_path,ai_type,ai_uuid,msg)
        #msg = str(out_dic)
        #write_msg(log_path,ai_type,ai_uuid,msg)
    except Exception as e:
        print(e)
        #print("f5eMatch post result error")
        #print(r.text)
        #print(r.status_code)
        #msg = 'f5eMatch requeests post error'
        write_msg(log_path,ai_type,ai_uuid,e)   

def get_match_topK(fea,dic,topK):
    sim={}
    for i in range(len(dic)):
        f2=list(dic.values())[i]
        key=list(dic.keys())[i]

        sim0 = np.dot(fea, f2.T)
        sim[key]=sim0
        
    predictions = sorted(sim.items(),key = lambda x:x[1],reverse = True)
    res = []
    for i in range(topK):
        d=str(i)
        d={}
        name=predictions[i][0].replace('.jpg','')
        si=predictions[i][1]
        d.update(labels=name)
        d.update(similarity=si)
        res.append(d)

    return res
################################################################################################

def main(old_db_list):
    
    input_url,output_url,log_path,ai_uuid = read_config(config_path)
    print(input_url)
    print(output_url)
    print(log_path)
    print("uuid:",ai_uuid)
    #input_url = 'http://14.152.78.59:9090/f5eMatchQueue/popTask'
    #output_url = ''
    msg='read config success.'
    write_msg(log_path,ai_type,ai_uuid,msg)
    
    #load face lib
    face_lib_dic = []
    try:
        start = time.time()
        with open('feature_new.pkl','rb+') as f:
            face_lib_dic = pickle.load(f)
        print('load face lib time:',time.time()-start)     
    except Exception as e:
        print(e)
    
    while(1):
        #time.sleep(5)
        tsb = time.time()
        ai_status = 1
        if read_state(job_path):
        #if True:   
            
            taskJson_dic, errorCode, errorMsg = read_input(input_url)
            #print(taskJson_dic)
            if len(taskJson_dic)!=7:
                msg = 'no f5eMatch job.'
                write_msg(log_path,ai_type,ai_uuid,msg)
                continue
            
            fea,db_list = read_fea_from_taskJson(taskJson_dic,tsb)
            #print("fea dim:")
            #print(fea.shape)
            #print(db_list[0]['num'])
            # read type and number for element db_list
            db_info_list = read_db_params(db_list)
            
            try:
                if len(face_lib_dic)<1:
                    msg = 'face lib is empty'
                    print(msg)
                    write_msg(log_path,ai_type,ai_uuid,msg)
                    continue
                else:
                    db_params = db_info_list[0]
                    topk = db_params['top']
                    topk_res = get_match_topK(fea,face_lib_dic,topk)
                    msg = 'get topk res.'
                    write_msg(log_path,ai_type,ai_uuid,msg)
                    print(msg)
                    
                    
                    
                    f5ematch_output(taskJson_dic,output_url,db_params,topk_res,log_path)
                    nums = len(topk_res)
                    write_logs(log_path,ai_type,ai_uuid,ai_status,taskJson_dic,nums,tsb,time.time())
                    print('f5eMatch success.')     
                    
            except Exception as e:
                    print(e)
                    msg = 'f5eMatch process failed.'
                    ai_status = -1
                    write_msg(log_path,ai_type,ai_uuid,e)
                    write_logs(log_path,ai_type,ai_uuid,ai_status,taskJson_dic,0,tsb,time.time())                
            
        else:
             taskJson_dic = []
             ai_status = 2
             #write_logs(log_path,ai_type,ai_uuid,ai_status,taskJson_dic,0,tsb,time.time())
             time.sleep(1)
             msg = ai_type +' is sleep.' 
             write_msg(log_path,ai_type,ai_uuid,msg)
  
    
if __name__ == '__main__':
    main(old_db_list)
