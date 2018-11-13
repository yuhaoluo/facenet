#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:49:25 2018

@author: luoyuhao
"""
import math  
import numpy as np
class Vector(object):  
    """docstring for Vector"""  
    """根据坐标轴列表输入 创建向量, 并创建该向量所处的空间维度"""  
    def __init__(self, coordinates):  
        super(Vector, self).__init__()  
        try:  
            if not coordinates:  
                raise ValueError  
            self.coordinates = tuple(coordinates)  
            self.dimension = len(coordinates)      
        except ValueError:  
            raise ValueError('The coordinates must be nonempty')  
        except TypeError:  
            raise TypeError('The coordinates must be an iterable')  
  
    # '''能够使python的内置print函数 输出向量坐标轴'''  
  
    def __str__(self):  
        return 'Vector: {}'.format(self.coordinates)  
  
    def __eq__(self, v):  
         return self.coordinates == v.coordinates  
      
    # 计算向量长度  
    def calculateSize(self):  
        result = 0  
        num = len(self.coordinates)  
        for i  in range(num):  
            result +=self.coordinates[i] * self.coordinates[i]  
        result = math.sqrt(result)  
        return round(result,8)  
  
    # 将向量归一化  
    def standardizaiton(self):  
        size = self.calculateSize()  
        new_corrdinate = [round(x/size,8) for x in self.coordinates]  
        #return Vector(new_corrdinate)
        return np.array(new_corrdinate)
  
myVector = Vector([-0.221,7.437])  
myVector2 = Vector([5.581,-2.136])  
myVector3 = Vector([8.813,-1.331,-6.247])  
myVector4 = Vector([1.996,3.108,-4.554])  
