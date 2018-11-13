#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 17:39:09 2018

@author: luoyuhao
"""
import json
import numpy as np
import time
import heapq
from vector_normalization import Vector

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
# =============================================================================
#     def __cmp__(self,other):
#         #call global(builtin) function cmp for int
#         return cmp(self.similarity,other.similarity)
# =============================================================================
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
        
####################################################################################################### 
face_json_path = "/home/luoyuhao/Datasets/ai_cloud_evaluate/json_lib/whiteList_256_20181102_1.json" 
start = time.time()      
labels,features = read_labels_and_features_from_json(face_json_path)
faces_lib_dic = create_face_lib(labels,features)
print('read face lib time:',time.time()-start)

test_features_path = '/home/luoyuhao/Datasets/ai_cloud_evaluate/test_features.npy'
test_labels_path = '/home/luoyuhao/Datasets/ai_cloud_evaluate/act_labels.npy'
test_features = np.load(test_features_path)
act_labels = np.load(test_labels_path)
print('load test features and labels.')
topK = 5

# =============================================================================
# faces_lib_dic_norm = faces_lib_dic.copy()
# for i in range(len(faces_lib_dic)):
#     tmp = faces_lib_dic[i]['feature'].tolist()
#     tmp_fea = Vector(tmp)
#     faces_lib_dic_norm[i]['feature'] = tmp_fea.standardizaiton()
#     
# test_features_norm = test_features.copy()
# for i in range(len(test_features_norm)):
#     tmp_fea = Vector(test_features[i].tolist())
#     test_features_norm[i]  = tmp_fea.standardizaiton()
# =============================================================================
    
#print(test_features[0].dtype)
    
    
face_cmp_res = []
for i in range(len(act_labels)):
    start = time.time()
    topk_res = get_topK(np.array(test_features[i]),faces_lib_dic,topK)
    print('\nmatch time:',time.time()-start)
    print('subject {} top {} res:'.format(act_labels[i],topK))
    for j in range(len(topk_res)):
        print('rank {} label:{}, similarity:{}'.format(j+1,topk_res[j].label,topk_res[j].similarity[0])) 
    face_cmp_res.append(topk_res)
    



