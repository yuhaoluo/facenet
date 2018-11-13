#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 15:49:11 2018

@author: luoyuhao
"""
import numpy as np
from scipy import misc
import cv2
import os

import imageio

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  


img_path = '/home/luoyuhao/Datasets/CommanTest/1.png';
img  = imageio.imread(img_path)
img_nor = (img-127.5)*0.0078125


# =============================================================================
# img_trans = img.transpose(1,0,2)
# img_path2 = '/home/luoyuhao/Datasets/CommanTest/1_trans.png'
# imageio.imwrite(img_path2,img_trans)
# =============================================================================

#cv2.namedWindow('hi',cv2.WINDOW_NORMAL)
#cv2.imshow('img',img)





# ========================picture prewhiten=====================================================
# img_path = '/home/luoyuhao/Datasets/CommanTest/1.png';
# print(img_path)
# img = imageio.imread(os.path.expanduser(img_path))
# #img = cv2.imread(os.path.expanduser(img_path))
# #img = misc.imread(os.path.expanduser(img_path))
# img_out = prewhiten(img)
# 
# =============================================================================
    
# =============================================================================
# 
# pnet_input = np.load("/home/luoyuhao/Datasets/CommanTest/stage2_input.npy")
# path = '/home/luoyuhao/Datasets/CommanTest/onet_input'
# 
# 
# file = open(path,'r')
# line = file.readline()
# a = line.split(',')
# res = []
# for i in a:
#     res.append(i)
#     
# 
# =============================================================================

# =============================================================================
# 
# for idx in range(1):
#     pixList = []
#     for i in range(24):
#         for j in range(24):
#            # pix = [round(pnet_input[idx,i,j,0],6), round(pnet_input[idx,i,j,1],6),pnet_input[idx,i,j,2]]
#             pixList.append(float('%.6f' % pnet_input[idx,i,j,0]))
#             pixList.append(float('%.6f' % pnet_input[idx,i,j,1]))
#             pixList.append(float('%.6f' % pnet_input[idx,i,j,2]))
#             
#             save_path = '/home/luoyuhao/Datasets/CommanTest/pic/img_{}.txt'.format(idx)
#             np.savetxt(save_path,pixList,fmt='%.6e',newline=',')
#     #print(pixList)
#     o = open(path,'a') 
#     for x in pixList:
#         o.write('%f,'%x)
#                 
# =============================================================================


