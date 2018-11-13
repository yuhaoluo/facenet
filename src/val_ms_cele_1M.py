#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:48:12 2018

@author: luoyuhao
"""
import os
import csv
import numpy as np
import shutil
class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    labels_string = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
        labels_string += [dataset[i].name]* len(dataset[i].image_paths)
    return image_paths_flat, labels_flat,labels_string

def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
  
    return dataset   #每个类及对应的图片路径

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def create_face_lib(path,lib_folder, has_class_directories=True):
    
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        
        face_list = os.listdir(os.path.join(path_exp, class_name))
        idx =  np.random.randint(len(face_list))
        print(face_list[idx])
        
        facedir = os.path.join(lib_folder, class_name)
        if not os.path.exists(facedir):
            os.mkdir(facedir)
        
        ori = os.path.join(path_exp, class_name)
        srcfile = os.path.join(ori,face_list[idx])
        dstfile = os.path.join(facedir,face_list[idx])
        shutil.move(srcfile,dstfile)
    return 




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
# =============================================================================
# 
# dataset_folder = '/home/luoyuhao/Datasets/TEST'
# dataset = get_dataset(dataset_folder)
# images,labels_num,labels_string = get_image_paths_and_labels(dataset)
# 
# =============================================================================

# =============================================================================
# fea_path = '/home/luoyuhao/Datasets/Embeding_test/feature.npy'
# fea = np.load(fea_path)
# =============================================================================


lib_features_path = '/home/luoyuhao/Datasets/ms-celeb-1M-evaluation/lib_features.npy'
lib_labels_path = '/home/luoyuhao/Datasets/ms-celeb-1M-evaluation/lib_labels.npy'

lib_fea = np.load(lib_features_path)
lib_labels = np.load(lib_labels_path)

test_features_path = '/home/luoyuhao/Datasets/ms-celeb-1M-evaluation/test_features.npy'
test_labesl_path = '/home/luoyuhao/Datasets/ms-celeb-1M-evaluation/act_labels.npy'

test_fea = np.load(test_features_path)
act_labels = np.load(test_labesl_path)

similarity_list = []
pre_label_list = []



for i in range(len(test_fea)):
    cur_fea = test_fea[i,:]
    sim_res = cosine_similarity(cur_fea,lib_fea)
    max_sim_idx = np.argmax(sim_res)
    pre_label_list.append(lib_labels[max_sim_idx])
    similarity_list.append(sim_res[max_sim_idx])
    
np.save('similarity.npy',similarity_list)    
np.save('pre_labels.npy',pre_label_list)

##                       Create   Face    Lib
# =============================================================================
# folder = '/home/luoyuhao/Datasets/ms-celeb-1M-evaluation/ms-test-data/data'  
# save_folder = '/home/luoyuhao/Datasets/ms-celeb-1M-evaluation/ms-test-data-lib'
# print("start create lib")
# create_face_lib(folder,save_folder)  
# 
# 
# =============================================================================

# =============================================================================
# with open('/home/luoyuhao/Datasets/ms-celeb-1M-evaluation/egg.csv', 'wb') as csvfile:
#     
#     spamwriter = csv.writer(csvfile,dialect='excel')
#     spamwriter.writerow([1,2,4])
# =============================================================================


