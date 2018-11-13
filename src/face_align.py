#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:04:39 2018

@author: luoyuhao
"""
import numpy
import numpy as np
import cv2
import time
import imageio
from skimage import transform as trans

imgSize = [112, 96];
coord5point = [[30.2946, 51.6963],
               [65.5318, 51.6963],
               [48.0252, 71.7366],
               [33.5493, 92.3655],
               [62.7299, 92.3655]]

face_landmarks = [[259, 137],
                  [319, 150],
                  [284, 177],
                  [253, 206],
                  [297, 216]]

def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return numpy.vstack([numpy.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),numpy.matrix([0., 0., 1.])])

def warp_im(img_im, orgi_landmarks,tar_landmarks):
    #pts1 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    #pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in tar_landmarks]))
    #M = transformation_from_points(pts1, pts2)
    orgi_landmarks = np.float64(np.matrix(orgi_landmarks))
    tar_landmarks = np.float64(np.matrix(tar_landmarks))
    M = transformation_from_points(orgi_landmarks, tar_landmarks)
    dst = cv2.warpAffine(img_im, M[:2], (160, 160))
    return dst

def face_align(img,bbox=None, landmark=None):
  #image_size = [np.array(img).shape[0],np.array(img).shape[1]]
  #assert len(image_size)==2
  #assert image_size[0]==160 and image_size[1]==160
  image_size = [160,160]
  if landmark is not None:
    assert len(image_size)==2
    std_mark = np.array([
        [54.7066,73.8519],
        [105.045,73.5734],
        [80.036,102.481],
        [59.3561,131.951],
        [101.043,131.72] ], dtype=np.float32 )

    face_mark = landmark.astype(np.float32)
    warped = warp_im(img,face_mark,std_mark)
    return warped

  else:
      if bbox is None: #use center crop
       
          det = np.zeros(4, dtype=np.int32)
          det[0] = int(img.shape[1]*0.0625)
          det[1] = int(img.shape[0]*0.0625)
          det[2] = img.shape[1] - det[0]
          det[3] = img.shape[0] - det[1]
      else:
          det = bbox
      margin = 32
      bb = np.zeros(4, dtype=np.int32)
      bb[0] = np.maximum(det[0]-margin/2, 0)
      bb[1] = np.maximum(det[1]-margin/2, 0)
      bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
      bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
      ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
      if len(image_size)>0:
          ret = cv2.resize(ret, (image_size[1], image_size[0]))
          return ret 

########################################################################################################################
def face_align2(img, bbox=None, landmark=None):

  M = None
  #image_size = [np.array(img).shape[0],np.array(img).shape[1]]
  #assert len(image_size)==2
  #assert image_size[0]==160 and image_size[1]==160
  image_size = [160,160]
  if landmark is not None:
    assert len(image_size)==2
    std_mark = np.array([
        [54.7066,73.8519],
        [105.045,73.5734],
        [80.036,102.481],
        [59.3561,131.951],
        [101.043,131.72] ], dtype=np.float32 )

    face_mark = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(face_mark, std_mark)
    M = tform.params[0:2,:]
  
  if M is None:
    if bbox is None: #use center crop
      det = np.zeros(4, dtype=np.int32)
      det[0] = int(img.shape[1]*0.0625)
      det[1] = int(img.shape[0]*0.0625)
      det[2] = img.shape[1] - det[0]
      det[3] = img.shape[0] - det[1]
    else:
      det = bbox
    margin = 44
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
    bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
    ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
    if len(image_size)>0:
      ret = cv2.resize(ret, (image_size[1], image_size[0]))
    return ret 
  else: #do align using landmark
    assert len(image_size)==2
    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
    #warped = warp_im(img,face_mark,std_mark)
    return warped





##################################################################################################################

# =============================================================================
# start = time.time()
# pts1 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in face_landmarks]))
# pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in coord5point]))
# M = transformation_from_points(pts1,pts2)
# 
# 
# 
# 
# img_path = '/home/luoyuhao/Datasets/Align/1.png'
# img = imageio.imread(img_path)
# =============================================================================


#def main():
# =============================================================================
#     pic_path = r'D:\20171117190537959.jpg'
#     img_im = cv2.imread(pic_path)
#     cv2.imshow('affine_img_im', img_im)
#     dst = warp_im(img_im, face_landmarks, coord5point)
#     cv2.imshow('affine', dst)
#     crop_im = dst[0:imgSize[0], 0:imgSize[1]]
#     cv2.imshow('affine_crop_im', crop_im)
# =============================================================================

# =============================================================================
# if __name__=='__main__':
#     main()
#     cv2.waitKey()
#     pass
# =============================================================================
