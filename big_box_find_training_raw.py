#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ultralytics import YOLO
import matplotlib.pyplot as p
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import cv2
import sys,os,glob
#%%  show results?
modelIn=path_to_/best.pt'
model1=YOLO(modelIn)
image_path =path_to_jpg 
results=model1(image_path)
print(results[0].boxes)
# results[0].show()
# xyxy=results[0].boxes.xyxy
# print(xyxy)
# x_min = xyxy[:, 0]  # First column: x_min
# y_min = xyxy[:, 1]  # Second column: y_min
# x_max = xyxy[:, 2]  # Third column: x_max
# y_max = xyxy[:, 3]
# xm=x_min.cpu().numpy()

xywh=results[0].boxes.xywh   #coordinates
conf1=results[0].boxes.conf    #confidence

coords=xywh.cpu().numpy()
conf=conf1.cpu().numpy()     #confidence
df1=pd.DataFrame(coords)
# df1.columns=['xc','yc','width','height']
df2=pd.DataFrame(conf)
df3=pd.concat([df1,df2],axis=1)
df3.columns=['xc','yc','width','height','conf']
print('df3.shape:',df3.shape)
df4=df3[df3.conf > 0.99]
print('df4.shape:',df4.shape)

## DWB @todo ##  print out instructions for using Aletheia to see
##           ##+ how well the boxes fit.
