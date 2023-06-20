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
import Jet_Contrast as JC
import utils 

import importlib
importlib.reload(utils) #Force Reload to ensure latest version is loaded
importlib.reload(JC) #Force Reload to ensure latest version is loaded

classes = ['LED','OG']  #Subclass that are compared
file_extension = '.wmv' #File extension of the video
subfolder = 'Data\\LED_Contrast\\' #Subfolder where videos are located

amount_of_runs = 1 #Amount of runs per class
transform = utils.transform(200,200,-70,250)

df = JC.contrast_analysis(amount_of_runs,
            classes,
            file_extension,
            subfolder,
            transform,
            show_frames=True,
            console_output=True)

#Use DataFrame Data to make plots
fig = sns.boxplot(data=df,x='Classes',y='Contrast')
fig.axes.set_title('Standard Deviation of Luminance',fontsize=24)
fig.set_xlabel('Class')
fig.set_ylabel('Standard Deviation of Luminance [cd/m^2]')
plt.tight_layout()






