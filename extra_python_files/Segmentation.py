#Open Source Libraries
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
import Jet_Segmentation as JS
import utils 

import importlib
importlib.reload(utils) #Force Reload to ensure latest version is loaded
importlib.reload(JS) #Force Reload to ensure latest version is loaded


file_extension = '.m4v' #File extension of the video
subfolder = 'Data\\LED_Contrast\\' #Subfolder where videos are located
filename = 'LED1'+file_extension
__location__ = os.path.abspath(os.path.join(os.path.dirname(__file__)))
filepath = __location__ +'\\'+ subfolder + filename

video = utils.vid(filepath)
frame = video.get_frame(1)
video.show_frame(frame)



