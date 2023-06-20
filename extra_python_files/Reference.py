# Open Source Libraries
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
import pyCompare as pyc
import importlib

#Custom Libraries
sys.path.insert(0, './lib')
import utils 
importlib.reload(utils)  #Force Reload to ensure latest version is loaded


#Dataframe with Microscopy Analysis
file_folder = "Data\\"
filename = 'Microscopy_Analysed.xlsx'
file_path = file_folder + filename

df_microscopy = pd.read_excel(file_path)
df_microscopy.set_index('Angle',inplace=True)
df_microscopy['type'] = 'Microscopy'
#print(df_microscopy)


# Dataframe with Manual Video Analysis
file_folder = "Data\\"
filename = 'Video_Analysed.xlsx'
file_path = file_folder + filename

df_video = pd.read_excel(file_path)
df_video.set_index('Angle',inplace=True)
df_video['type'] = 'Video'
#print(df_video)

merged_df = pd.concat([df_microscopy, df_video], axis=0)
print(merged_df)


# Reset the index to convert the angles into a column
df_reset = merged_df.reset_index()

# Reshape the data using melt to have F numbers as columns
df_melt = df_reset.melt(id_vars=['Angle', 'type'], var_name='Speed [mm/min]', value_name='Jet Lag [mm]')

# Create the box and whisker plot
sns.boxplot(x='Speed [mm/min]', y='Jet Lag [mm]', hue='type', data=df_melt)

# Define the desired x-axis labels
x_labels = range(100, 301, 50)

# Set the x-axis tick positions and labels
plt.xticks(range(len(x_labels)), x_labels)

plt.title('Positive and Negative Control of Jet Lag Quantification', fontsize=12, fontweight='bold')
