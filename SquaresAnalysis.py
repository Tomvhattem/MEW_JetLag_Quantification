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
import matplotlib.colors as mcolors

#Custom Libraries
sys.path.insert(0, './lib')
import utils 
import plots 
importlib.reload(utils)  #Force Reload to ensure latest version is loaded
importlib.reload(plots) 

#Dataframe with Microscopy Analysis
file_folder = "Data\\"
filename = 'Microscopy_Analysed.xlsx'
file_path = file_folder + filename

df_microscopy = pd.read_excel(file_path)
df_microscopy.set_index('Angle',inplace=True)
#print(df_microscopy)


# Dataframe with Manual Video Analysis
# file_folder = "Data\\"
# filename = 'Video_Analysed.xlsx'
# file_path = file_folder + filename

# df_video = pd.read_excel(file_path)
# df_video.set_index('Angle',inplace=True)
df_video = pd.read_csv('csv\\Manual.csv')
df_video.set_index('Angle',inplace=True)
"""             ALGORITHM                   """
#Algorithm
nr_of_videos = 8 

classes = ['F100','F150','F200','F250','F300'] 
folder = 'Data\\Squares\\Squares\\Cubes' #Subfolder where videos are located
Angles = ['Angle%i' % i for i in range(1,nr_of_videos+1)]
file_extension = ".mov"

nr_of_analyzed_frames = 5
motion_blur_buffer = 3

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
                annotate=False,
                threshold = False,
                CLI=False,
                error=False,
                jetlag = False,  #reference original frame
                CP=False
                )

df_algorithm_median = []
df_class = []
df_angle = []
df_nozzle_pixels = []
df_nozzle_centre = []
df_bottom_nozzle = []
df_CP_height = []

for subfolder in classes:
    for i in range(nr_of_videos):
        filename = Angles[i] + file_extension
        filepath = folder + "\\" + subfolder + "\\" + filename 

        JL = utils.Jet_Lag(filepath,
                calibration_frame='Centre',
                local_debug=debug,
                CP_calibrated_distance=CP_calibrated_distance,
                CP_mode=CP_mode,
                main_threshold = main_threshold
                )

        output = 'frames_output\\jet_lag\\'+str(subfolder)+"\\"+str(Angles[i])+"\\"
        JL.set_output_folder(output)
        last_frame = int(JL.video.total_frames) - motion_blur_buffer

        JL.video.set_calibration_frame(calibration_frame)
        first_frame = int(last_frame - motion_blur_buffer - nr_of_analyzed_frames)

        all_frames = range(first_frame,last_frame)
        _,*result = JL.find_jet_lag_video(
                                        frame_range=all_frames,
                                        return_extended_output=True
                                        )
        jet_lag_video = result[0]
        nozzle_pixels = int(result[1])
        nozzle_centre = int(result[2][0])
        bottom_nozzle_y = int(result[3])
        CP_height_pixels = int(result[4])

        #Usually 
        jetlag = max(jet_lag_video)
        if jetlag > 7.5:
            jetlag = np.median(jet_lag_video)
        df_algorithm_median.append(jetlag)

        df_class.append(subfolder)
        df_angle.append(Angles[i])

        df_nozzle_pixels.append(nozzle_pixels)
        df_nozzle_centre.append(nozzle_centre)
        df_bottom_nozzle.append(bottom_nozzle_y)
        df_CP_height.append(CP_height_pixels)

df_algorithm = pd.DataFrame({
    'Median_Jet_Lag': df_algorithm_median,
    'Classes': df_class,
    'Angle': df_angle,
    'Nozzle': df_nozzle_pixels,
    'Nozzle Centre': df_nozzle_centre,
    'Bottom Nozzle': df_bottom_nozzle,
    'Collector Plate Height': df_CP_height
})

df_algorithm_copy = df_algorithm.copy()
df_algorithm = df_algorithm_copy.pivot(index='Angle',columns='Classes',values='Median_Jet_Lag')
df_nozzle = df_algorithm_copy.pivot(index='Angle',columns='Classes',values='Nozzle')
df_nozzle_centre= df_algorithm_copy.pivot(index='Angle',columns='Classes',values='Nozzle Centre')
df_bottom_nozzle= df_algorithm_copy.pivot(index='Angle',columns='Classes',values='Bottom Nozzle')
df_CP_height= df_algorithm_copy.pivot(index='Angle',columns='Classes',values='Collector Plate Height')

# Compute the difference between the dataframes
diff_df = df_algorithm- df_video
#diff_video_microscopy = df_video-df_microscopy
df_video_copy = df_video.copy()

video = df_video.values.flatten().tolist()
#microscopy = df_microscopy.values.flatten().tolist()
algorithm = df_algorithm.values.flatten().tolist()

#df_microscopy['type'] = 'Microscopy'
df_video['type'] = 'Video'
df_algorithm['type'] = 'Algorithm'

merged_df = pd.concat([ df_video,df_algorithm], axis=0) #df_microscopy,

df_reset = merged_df.reset_index()
df_melt = df_reset.melt(id_vars=['Angle', 'type'], var_name='Speed [mm/min]', value_name='Jet Lag [mm]')

"""PLOTS"""
palette = sns.color_palette("mako")
ba_palette = sns.color_palette("mako", n_colors=3) 
hex_colors = [mcolors.rgb2hex(color) for color in ba_palette]

""""BLAND ALTMAN PLOT"""
nr_of_speeds = len(classes)
#plots.blandaltman_plot(microscopy,algorithm,'plots/blandaltman_micro_algo.png')
plots.blandaltman_plot(video,algorithm,'plots/blandaltman_video_algo.png',nr_of_speeds )
plots.blandaltman_plot_percentage(video,algorithm,'plots/blandaltman_video_algo_percentage.png',nr_of_speeds )

"""CUSTOM BOX AND WHISKERS PLOT"""
palette_custom = sns.color_palette("mako",3)
sns.boxplot(x='Speed [mm/min]', y='Jet Lag [mm]', hue='type', data=df_melt,palette=palette_custom)
x_labels = range(100, 301, 50)
plt.xticks(range(len(x_labels)), x_labels)
plt.title('Jet Lag Quantification', fontsize=18, fontweight='bold')
plt.savefig('plots\\Box_pos_neg_control_algorithm.png', dpi=400)
plt.show()


"""Large HEATMAP"""
fig, axes = plt.subplots(1, 5, figsize=(24,4))
heatmap_data = [(diff_df.abs(), 'Algorithm and Video Differences',"Differences [mm]"),
                (df_nozzle, 'Nozzle Diameter',"Nozzle Diameter [Pixels]"),
                (df_nozzle_centre, 'Nozzle Centre X Location',"Nozzle Centre X Location [Pixels]"),
                (df_bottom_nozzle, 'Nozzle Bottom Y Location',"Nozzle Bottom Y Location [Pixels]"),
                (df_CP_height, "Collector Plate Height","Collector Plate Height [Pixels]")]
y_labels = []
for run in range(1, 5):
    y_labels.append(f"Run {run} R")
    y_labels.append(f"Run {run} L")

for (data, title, colorbar_title), ax in zip(heatmap_data, axes.flatten()):
    sns.heatmap(data, annot=True, cmap=palette, fmt='.2f',ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Speed [mm/min]')
    ax.set_ylabel('Video Index')
    colorbar = ax.collections[0].colorbar
    colorbar.set_label(colorbar_title)

plt.tight_layout()
plt.savefig('plots/CombinedHeatMap.png', dpi=400)
plt.show()



plots.heatmap(diff_df,
            palette,
            'plots\\HeatMap.png',
            'Algorithm and Video Differences',
            "Difference [mm]")

df_algorithm_plot = df_algorithm.copy()
df_algorithm_plot.drop('type', axis=1, inplace=True)
plots.heatmap(df_algorithm_plot,
            palette,
            'plots\\HeatMapabsolutemeasurement.png',
            'Algorithm Measurements',
            "Jet Lag [mm]")

plots.heatmap(df_nozzle,
            palette,
            'plots\\HeatMapNozzle.png',
            'Nozzle Diameter',
            "Nozzle Diameter [Pixels]")

plots.heatmap(df_nozzle_centre,
            palette,
            'plots\\HeatMapNozzleCentre.png',
            'Nozzle Centre X Location',
            "Nozzle Centre Location [Pixels]")

plots.heatmap(df_bottom_nozzle,
            palette,
            'plots\\HeatMapNozzleBottom.png',
            'Nozzle Bottom Y Location',
            "Bottom Nozzle Y Location [Pixels]")

plots.heatmap(df_bottom_nozzle,
            palette,
            'plots\\HeatMapCP.png',
            'Collector Plate Height',
            "Collector Plate Height [Pixels]")

# plots.heatmap(diff_video_microscopy,
#             palette,
#             'plots\\HeatMapVideoMicroscopy.png',
#             'Difference Video Microscopy',
#             "Difference Video Microscopy [mm]")



plots.boxplot(diff_df,
    palette,
    'plots\\Box_differences_percentage.png',
    'Algorithm and Video Differences',
    'Speed [mm/min]',
    'Differences [mm]',
    )
    
plots.boxplot((diff_df / df_video_copy *100),
    palette,
    'plots\\Box_differences.png',
    'Percentual Differences',
    'Speed [mm/min]',
    'Deviation [%]',
    )

df_melt.loc[df_melt['type'] == 'Video', 'type'] = 'Manual'
"""Create Box and Whiskers Plot of jetlag with respect to CTS"""
palette_custom = sns.color_palette("mako",3)
sns.boxplot(x='Speed [mm/min]', y='Jet Lag [mm]', hue='type', data=df_melt,palette=palette_custom)
plt.title('Manual - Algorithm Comparison', fontsize=18, fontweight='bold')
plt.xlabel('CTS')
plt.ylabel('Jet Lag [mm]')
plt.legend(title="Quantification Method")
xtick_labels = ["1x", "1.5x", "2x", "2.5x", "3x"]
plt.xticks(range(len(xtick_labels)), xtick_labels)
plt.savefig('plots\\Box_CTS.png', dpi=400)
plt.show()



"""Data Processing for next plot"""
df_algo_reset = df_algorithm.reset_index()
df_melt_algorithm = df_algo_reset.melt(id_vars=['Angle', 'type'], var_name='Speed [mm/min]', value_name='Jet Lag [mm]')
df_melt_algorithm['Speed [mm/min]'] = df_melt_algorithm['Speed [mm/min]'].str.extract('(\d+)').astype(int)

"""Create Line Plot of jetlag with respect to CTS"""
# Create the box and whisker plot
sns.regplot(x='Speed [mm/min]', y='Jet Lag [mm]', data=df_melt_algorithm,order=2,ci=None,color=palette[3])
plt.title('Jet Lag Quantification', fontsize=18, fontweight='bold')
plt.xlabel('CTS')
plt.ylabel('Jet Lag [mm]')
xtick_labels = ["",'1x', "",'1.5x',"", '2x', "",'2.5x', "",'3x']
ax = plt.gca()
ax.set_xticklabels(xtick_labels)
plt.savefig('plots\\Line_CTS.png', dpi=400)
plt.show()


