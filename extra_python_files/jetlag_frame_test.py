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

file_extension = '.mp4' #File extension of the video
subfolder = 'Data\\Squares\\Squares\\Cubes\\old.reference\\FullVideos\\' #Subfolder where videos are located
angle = 'F300'
filename = angle+file_extension
__location__ = os.path.abspath(os.path.join(os.path.dirname(__file__)))
filepath = __location__ +'\\'+ subfolder + filename

nr_of_analyzed_frames = 2
motion_blur_buffer = 0

CP_calibrated_distance = 2.85#mm
CP_mode = "Calibration"

debug = utils.debug(
                segmentation=False, #only used in legacy rotation
                scharr_segmentation=False, #Top Segmentation
                calibration=False,
                rotation=False,
                search_height = False,
                height_threshold = False,
                annotate=False,
                threshold = False,
                CLI=False,
                error=False,
                jetlag = False  
                )


JL = utils.Jet_Lag(filepath,
                calibration_frame='Start',
                local_debug=debug,
                CP_calibrated_distance=CP_calibrated_distance,
                CP_mode=CP_mode
                )

output = 'frames_output\\jet_lag_full\\'+str(subfolder)+"\\"+str(angle)+"\\"
JL.set_output_folder(output)

last_frame = int(JL.video.total_frames) - motion_blur_buffer
first_frame = int(last_frame - motion_blur_buffer - nr_of_analyzed_frames)

all_frames = range(first_frame,last_frame)
_,jet_lag_video = JL.find_jet_lag_video()#frame_range=all_frames
print(_)
print(jet_lag_video)


