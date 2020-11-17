#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:03:57 2020

@author: MoHan
"""

# libraries needed
import os

import math
from network import Net
import matplotlib.pyplot as plt
import numpy 

# note: if tensorflow is not install, run "pip install --upgrade tensorflow"
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image
import random

test_dir = "./test_dataset/test_set"
train_dir = "./test_dataset/training_set"

train_dir_cats = train_dir + "/cats"
train_dir_dogs = train_dir + "/dogs"
test_dir_cats = test_dir + "/cats"
test_dir_dogs = test_dir + "/dogs"

train_data = []
train_data_label = []
test_data = []
test_data_label = []

# Only transformed to gray pic
def normal_transform (imgpath):
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (28,28))
    return Image.fromarray(img)

# Preprocessed using gaussian_canny
def gaussian_canny_transform (imgpath):
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gaussian = cv2.GaussianBlur(img, (3,3), 0)
    gaussian = gaussian.astype(numpy.uint8)
    canny = cv2.Canny(gaussian, 50, 50)
    canny = cv2.resize(canny, (28,28))
    return Image.fromarray(canny)

# Preprocessed using sobel
def sobel_transform (imgpath):
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, weight = img.shape
    sobel = numpy.zeros((height, weight, 1), numpy.uint8)
    for i in range(0,height-2):
        for j in range(0,weight-2):
            gy=img[i,j]*1+img[i,j+1]*2+img[i,j+2]*1-img[i+2,j]-2*img[i+2,j+1]-img[i+2,j+2]*1
            gx=img[i,j]*1-img[i,j+2]+img[i+1,j]*2-2*img[i+1,j+2]+img[i+2,j]-img[i+2,j+2]
            grad=math.sqrt(gx*gx+gy*gy)
            if grad>50:
                sobel[i,j]=255
            else:
                sobel[i,j]=0
    print(sobel)
    # return Image.fromarray(numpy.uint8(sobel))

# Reading training data
def read_training_data(train_data, train_data_label, dir, label):
    for filename in os.listdir(dir):
        imgpath = dir + "/" + filename
        img = normal_transform(imgpath)
        train_data.append([numpy.asarray(img)])
        train_data_label.append((label))

# Reading testing data
def read_testing_data(test_data, test_data_label, dir, label):
    for filename in os.listdir(dir):
        imgpath = dir + "/" + filename
        img = normal_transform(imgpath)
        test_data.append([numpy.asarray(img)])
        test_data_label.append((label))

#read gray images into train_data and train_data_label
read_training_data(train_data, train_data_label, train_dir_cats, (0,1,0,0,0,0,0,0,0,0))
read_training_data(train_data, train_data_label, train_dir_dogs, (1,0,0,0,0,0,0,0,0,0))

#read gray images into test_data and test_data_label
read_testing_data(test_data, test_data_label, test_dir_cats, (0,1,0,0,0,0,0,0,0,0))
read_testing_data(test_data, test_data_label, test_dir_dogs, (1,0,0,0,0,0,0,0,0,0))
train_data = numpy.array(train_data)
test_data = numpy.array(test_data)
train_data_label = numpy.array(train_data_label)
test_data_label = numpy.array(test_data_label)

print(train_data.shape, train_data_label.shape)
print(test_data.shape, test_data_label.shape)
LeNet = Net()

print('Training Lenet......')
LeNet.train(training_data=train_data,training_label=train_data_label,batch_size=5,epoch=1,weights_file="pretrained_weights.pkl")

print('Testing Lenet......')
LeNet.test(data=test_data,label=test_data_label,test_size=25)

print('Testing with pretrained weights......')
LeNet.test_with_pretrained_weights(test_data, test_data_label, 25, 'pretrained_weights.pkl')