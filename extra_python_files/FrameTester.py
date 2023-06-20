import numpy as np
import cv2 as cv2
import time
import datetime
import math
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import seaborn as sns
import sys

#Custom Libraries
sys.path.insert(0, './lib')
import Jet_Contrast as JC
import Jet_Vibration_Analysis as JVA
import utils 

import importlib
importlib.reload(utils) #Force Reload to ensure latest version is loaded
importlib.reload(JC) #Force Reload to ensure latest version is loaded
importlib.reload(JVA)  #Force Reload to ensure latest version is loaded

file_extension = '.wmv' #File extension of the video
subfolder = 'Data\\LED_Contrast\\' #Subfolder where videos are located
filename = 'OG1'+file_extension
__location__ = os.path.abspath(os.path.join(os.path.dirname(__file__)))
filepath = __location__ +'\\'+ subfolder + filename

video = utils.vid(filepath)
frame = video.get_frame(150)

mm_per_pixel, nozzle_diameter_pixels,left_edge_nozzle,right_edge_nozzle = video.nozzle_calibration()
print('Nozzle Diameter ',nozzle_diameter_pixels)
x_centre_nozzle = (left_edge_nozzle + right_edge_nozzle) /2 
print('X_centre nozzle ',x_centre_nozzle)

search_height,_ = video.find_search_height(show_frames=True)
print('search height: ',search_height)


# video.show_frame(frame)

# frame_gray = video.convert_grayscale(frame)
# video.show_frame(frame)

# ret,threshold = cv2.threshold(frame_gray,70,255,cv2.THRESH_BINARY)
# video.show_frame(threshold)


# ada_threshold_image = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 9)
# video.show_frame(ada_threshold_image)


# kernel = np.ones((2, 2), np.uint8)
# closed_image = cv2.morphologyEx(ada_threshold_image, cv2.MORPH_CLOSE, kernel)
# closed_image = 255 - closed_image
# video.show_frame(closed_image)

# kernel = np.ones((12,12), np.uint8)
# dilated_edges = cv2.dilate(closed_image, kernel, iterations=1)
# video.show_frame(dilated_edges)

# combined_image = cv2.bitwise_and(threshold, dilated_edges)
# combined_image_color = video.convert_color(combined_image)
# video.show_frame(combined_image_color)


# #combined_image = video.rotate_frame(combined_image)

# coordinates = (int(x_centre_nozzle),int(search_height))
# coordinates = video.rotate_coordinates(coordinates)
# marked_image= video.add_circle(combined_image_color,coordinates,(255,0,0))

# video.show_frame(marked_image)
    
# y= int(2*video.height/3)
# CP_height = video.find_y_edge(combined_image,20,y,direction='Down')


# NP = (x_centre_nozzle,CP_height)
# print('NP', NP)

# marked_image= video.add_circle(marked_image,NP,(0,255,0))
# video.show_frame(marked_image)

# JCP_x = video.find_x_edge(combined_image,int(x_centre_nozzle),int(CP_height),direction='Left')
# print('JCP x', JCP_x)

# JCP = (JCP_x,CP_height)
# marked_image= video.add_circle(marked_image,JCP,(0,0,255))
# video.show_frame(marked_image)

# JetLag = abs(JCP_x - x_centre_nozzle)*mm_per_pixel
# print(round(JetLag,2), "[mm]")