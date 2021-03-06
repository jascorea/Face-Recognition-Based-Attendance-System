import pickle
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import os

imageset=[]
cap = cv2.VideoCapture(0)
count=0
while(True):
	ret, frame = cap.read()
	cv2.imshow('frame',frame)
		#if cv2.waitKey(1) & 0xFF == ord('q'):
	if(count==4):
		break
	cv2.imshow('frame',frame)
	print(type(frame))
	resized=cv2.resize(frame,(96,96))
	imageset.append(resized)
	count=count+1
imageset2=np.array(imageset)
print(imageset2.shape)
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("model loaded")
img_to_encoding(imageset2, FRmodel)
print("create encding")