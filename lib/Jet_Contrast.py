import numpy as np
import cv2 as cv2
import time
import datetime
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import warnings

import utils

def std_contrast(frame,console=False):
    """Calculate the standard deviation of the luminance, a metric for 
    contrast, for a given frame.

    Args:
        frame (Image Array): OpenCV BGR Frame object
        console (bool, optional): Toggle for console print. Defaults to False.

    Returns:
        contrast (Float): standard deviation of the luminance
    """    
    lum = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    std_dev = np.std(lum)
    contrast = std_dev
    if console: print("Contrast:", contrast)
    return float(contrast)

def average_contrast(video,show_frames=False):
    """Calculates the standard deviation of the luminance using the 
    std_contrast function for all frames of a video. Returns the average.

    Args:
        video (Vid Class Object): Video from the utils library.

    Returns:
        Average Contrast (Float): Contrast of all frames divided by the amount of frames.
    """    


    last_frame_id = video.get_duration()
    accumulated_contrast = 0
    for frame_id in range(0,last_frame_id-1):
        
        video.set_current_frame(frame_id)
        cropped_frame = video.crop_current_frame()
        
        accumulated_contrast += std_contrast(cropped_frame)
    if show_frames: 
        video.show_frame(cropped_frame)
        video.plot_histogram(cropped_frame)
    return accumulated_contrast/last_frame_id
    


def contrast_analysis(amount_of_runs,
            classes,
            file_extension,
            subfolder,
            transform=0,
            show_frames=False,
            console_output=False):
    """_summary_

    Args:
        amount_of_runs (_type_): _description_
        classes (_type_): _description_
        file_extension (_type_): _description_
        subfolder (_type_): _description_
        transform (_type_): _description_
        show_frames (bool, optional): _description_. Defaults to False.
        console_output (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """                    
    df_classes=[]
    df_contrast=[]
    df_run=[]
    for prefix in classes:
        for run in range(1,(int(amount_of_runs)+1)):
            filenames = []
            speed_list = []
            run_name = str(run)

            filename = prefix+str(run_name)+file_extension
            __location__ = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

            filepath = __location__ +'\\'+ subfolder + filename

            if os.path.exists(filepath):
                video = utils.vid(filepath)
                if transform != 0: video.set_transform(transform)
                local_contrast = average_contrast(video,show_frames)
                if console_output: print(local_contrast)

                if prefix == 'OG': df_classes.append('Original')
                else: df_classes.append(prefix)
                df_contrast.append(local_contrast)
                df_run.append(run)
            else:
                warnings.warn("Could not find video at "+str(filepath))
    df = pd.DataFrame()
    df['Classes'] = df_classes
    df['run'] = df_run
    df['Contrast'] = df_contrast

    return df