"""
TU/e
Tom van Hattem - 1580698
Biomaterials Processing and Design

Script for analyzing MEW Nozzle Blurring
Expects a series of videos where two classes can be compared (also works for 1 class)
Automatically detects where the printhead start to move
by analyzing the frame and checking for any movement in the frame.
Calculates the amount of blur in the image by subtracting the first non blurred
image from a range of blurred images. Accumulates the thresholded 'blur' in the range

percentage of blur for this accumulated white pixel image is calculated by dividing by 
the total pixel count. This is a relative quantification.

returns a dataframe with the class, speed and percentage of the blur.
This data can then be plotted. 

Blurr detection algorithm is found in utils. 

Please use a constant background for consitant results.

For plottin the frames ipython is used, do not run directly in normal console svp.
"""
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

#Custom Libraries
sys.path.insert(0, './lib')
import Jet_Vibration_Analysis as JVA
import utils 

import importlib
importlib.reload(utils)  #Force Reload to ensure latest version is loaded
importlib.reload(JVA)  #Force Reload to ensure latest version is loaded

"""VARIABLES"""
#Video should be defined as Class_run_VariableSpeed
#For example: New_1_F1000.wmv

classes = ['Og','ABS','SLS','PLA']  #Subclass that are compared 
#classes = ['SLS','SLS_2.5cm_']  #Subclass that are compared 
variable = 'F' #The variable that is experimented with
file_extension = '.wmv' #File extension of the video
subfolder = 'Data\\Vibration_3\\' #Subfolder where videos are located

amount_of_runs=4 #Runs per class

variable_start = 100 #Start of speed range
variable_end = 1000 #End of speed range
variable_step = 100 #Stepsize of speed range

vibration_threshold = 80 #Threshold used, higher this if percentages are high, inspect frames_output for more information on the threshold


#strongly advised not to plot too many images, as it gets slow
debug = utils.debug(
                segmentation=False, #only used in legacy rotation
                scharr_segmentation=False, #Top/Nozzle Segmentation
                calibration=False,
                rotation=False,
                search_height = False,
                height_threshold = False,
                annotate=False,
                threshold = False,
                CLI=False,
                blur=False
                )

df = JVA.blur_analysis_series(amount_of_runs,
                classes,
                variable,
                file_extension,
                variable_start,
                variable_end,
                variable_step,
                subfolder,
                vibration_threshold,
                debug,
                mode="Full" #Full or LinearOnly
                )



df.to_csv('csv\\Vibrations4.csv',index=False)

############################ DATA ANALYTICS ###################################

#df.loc[df['Classes'] == 'Original', 'Nozzle Diameter in Pixels'] = 24

mean = df['Nozzle Diameter in Pixels'].mean()
std = df['Nozzle Diameter in Pixels'].std()
threshold = mean + 2 * std

df['Nozzle_no_outliers'] = np.where(df['Nozzle Diameter in Pixels'] > threshold, np.nan, df['Nozzle Diameter in Pixels'])

nozzle_averages = df.groupby('Classes')['Nozzle_no_outliers'].transform('mean')

df['Normalized Blur'] = df['Blur'] / nozzle_averages  * 100

palette = sns.color_palette("mako", 4)
fig = sns.lineplot(data=df, x="Speed", y="Normalized Blur", hue="Classes",palette=palette,estimator='median')
plt.legend(loc = "upper left",title='Material')
fig.axes.set_title('Material-Specific Pixel Change per Speed', fontsize=18, fontweight='bold')
fig.set_xlabel('Speed [mm/min]')
fig.set_ylabel('Normalized Pixel Change [-]')
plt.savefig('plots\\vibrations_lineplot2.png',dpi=400)
plt.show()
plt.clf()

palette = sns.color_palette("mako", 4)
fig = sns.barplot(data=df, x="Speed", y="Normalized Blur", hue="Classes",palette=palette,estimator='median')
plt.legend(loc="upper left", title='Material')
plt.title('Material-Specific Pixel Change per Speed', fontsize=18, fontweight='bold')
plt.xlabel('Speed [mm/min]')
plt.ylabel('Normalized Pixel Change [-]')
plt.savefig('plots\\vibrations_barplot2.png', dpi=400)
plt.show()
plt.clf()



"""HEATMAP"""
unique_classes = df['Classes'].unique()
# Determine the number of rows and columns for the subplots
num_rows = 1  # Adjust as needed
num_cols = 4  # Adjust as needed

# Create the subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3))

# Flatten the axes array to simplify indexing
axes = axes.flatten()

# Iterate over the unique classes and create heatmaps
for i, class_name in enumerate(unique_classes):
    # Filter the DataFrame for the current class
    class_data = df[df['Classes'] == class_name]

    # Pivot the class data for the heatmap
    heatmap_data = class_data.pivot("Speed", "run", "Normalized Blur")

    # Select the appropriate subplot for the current class
    ax = axes[i]

    # Create the heatmap for the current class
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=palette, ax=ax)

    # Set the title of the subplot
    ax.set_title(class_name)

# Adjust the layout and spacing of the subplots
plt.tight_layout()
plt.savefig('plots\\vibrations_heatmap.png', dpi=400)

# Show the plot
plt.show()

# %%
