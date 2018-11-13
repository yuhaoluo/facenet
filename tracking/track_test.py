#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:44:30 2018

@author: luoyuhao
"""

import cv2
import imageio
video_path = "/home/luoyuhao/Datasets/Tracking/video/2.mkv"
save_path = "/home/luoyuhao/Datasets/Tracking/video/1.jpg"
video = cv2.VideoCapture(video_path)
ok,frame = video.read(0)
imageio.imwrite(save_path,frame)