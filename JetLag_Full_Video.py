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
output_folder = 'Data\\Frames2\\' #Subfolder where videos are written
filename = 'OG1'+file_extension
__location__ = os.path.abspath(os.path.join(os.path.dirname(__file__)))
filepath = __location__ +'\\'+ subfolder + filename

JL = utils.Jet_Lag(filepath,CLI_initialisation=True,show_frames=True)

JL.set_output_folder(output_folder)

JL.find_jet_lag_frame(100,CLI=True,show_frame=True)


all_frames,jet_lag_video = JL.find_jet_lag_video()



df = pd.DataFrame({'all_frames': all_frames, 'jet_lag_video': jet_lag_video})
df = df.mask(df['jet_lag_video'] > 2)
df_rolling = df.rolling(3, center=True).mean()

df['jet_lag_video']=df['jet_lag_video'].where(df_rolling['jet_lag_video'].notna(),np.nan)

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(ax=ax,
            x=all_frames,
            y=jet_lag_video,
            data=df,
            hue=df["jet_lag_video"].isna().cumsum(), 
            palette=["black"]*sum(df["jet_lag_video"].isna()), 
            legend=False, 
            markers=True)

plt.ylim(0, 4)
ax.set_title('Jet Lag',fontsize=24)
ax.set_xlabel('Frame Number [-]')
ax.set_ylabel('Jet Lag [mm]')