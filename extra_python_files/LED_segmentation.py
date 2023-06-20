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

CP_calibrated_distance = 3#mm
CP_mode = "Calibration"

calibration_frame = "End" #options: Start, Centre or End
main_threshold = 75 #was 70

#strongly advised not to plot too many images, as it gets slow
debug = utils.debug(
                segmentation=False, #only used in legacy rotation
                scharr_segmentation=False, #Top/Nozzle Segmentation
                calibration=False,
                rotation=False,
                search_height = False,
                height_threshold = False,
                annotate=True,
                threshold = False,
                CLI=True,
                blur=False
                )

# file_extension = '.wmv' #File extension of the video
# subfolder = 'Data\\LED_Contrast\\' #Subfolder where videos are located
# output_folder = 'Data\\Frames2\\' #Subfolder where videos are written
# filename = 'OG1'+file_extension

file_extension = '.wmv' #File extension of the video
subfolder = 'Data\\Height\\Large Distance 10cm LED ON\\' #Subfolder where videos are located
output_folder = 'Data\\Frames3\\' #Subfolder where videos are written
filename = 'Orthogonal_Nozzle_Run1'+file_extension

__location__ = os.path.abspath(os.path.join(os.path.dirname(__file__)))
filepath = __location__ +'\\'+ subfolder + filename

JL = utils.Jet_Lag(filepath,
            calibration_frame='End',
            local_debug=debug,
            CP_calibrated_distance=CP_calibrated_distance,
            CP_mode=CP_mode,
            main_threshold = main_threshold
            )

JL.set_output_folder(output_folder)

JL.find_jet_lag_frame(490)