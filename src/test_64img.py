#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:15:30 2018

@author: luoyuhao
"""
import cv2
import imageio
import numpy as np
import base64
import tensorflow as tf
print(tf.__version__)


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

def array64_to_base64(array):
    return base64.b64encode(array)

def base64_to_array64(base64_code):
    data = base64.b64decode(base64_code)
    array = np.frombuffer(data, dtype=np.float64)
    return array


# =============================================================================
# img_path = "/home/luoyuhao/Datasets/Docker/1.png"
# img_io = imageio.imread(img_path)
# encode_img = image_to_base64(img_io)
# decode_img = base64_to_image(encode_img)
# print(decode_img.shape)
# =============================================================================


arr = 0.5 * np.ones(256)
print(arr.dtype)
print(arr.shape)

#arr_str = str(arr)
#str_arr = arr_str.split()
#str_arr[0] = str_arr[0].split('[')[1]
#str_arr[-1] = str_arr[-1].split(']')[0]
#str_float = float(str_arr)

b64code = array64_to_base64(arr)
decode_arr = base64_to_array64(b64code)

print("decode res:")
print(np.allclose(arr,decode_arr ))