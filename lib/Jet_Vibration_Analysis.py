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

def get_moving_frame(vid,threshold=75):
    """Finds the first frame where there is movement, calculated by thresholding the difference between 
    two frames. And dilation of the result. If there is a contour in the dilation it return the frame_id
    
    Args:
        vid (Class Object): video class from Utils
        threshold (int, optional): threshold value. Defaults to 25.

    Returns:
        frame_id (int): first frame that contains movement
    """    
    # Initialize variables
    frame_id = 0
    found_moving_frame = False

    # Read the first frame
    ret, prev_frame = vid.video.read()

    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        # Read the next frame
        ret, curr_frame = vid.video.read()
        frame_id += 1
        
        # If there is no more frame, break the loop
        if not ret:
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Calculate the absolute difference between the current frame and the previous frame
        diff = cv2.absdiff(curr_gray, prev_gray)

        # Apply thresholding to the difference image
        thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

        # Dilate the thresholded image to fill in the holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=4)

        # Find contours in the dilated image
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if there is any contour with a sufficient area
        for contour in contours:
            if cv2.contourArea(contour) > 5000:
                found_moving_frame = True
                break
        
        # If a moving frame is found, break the loop and return the frame ID
        if found_moving_frame:
            break
        
        # Update the previous frame
        prev_gray = curr_gray

    # Release the video file and return the frame ID
    #vid.release()
    cv2.destroyAllWindows()
    return frame_id

def count_white(img,nozzle_diameter_pixels,threshold=60):
    """Converts image to gray scale,thresholds the image and count non zero pixels. 

    Args:
        img (Image Array): OpenCV image
        threshold (int,optional): value used for threshold function

    Returns:
        WhitePixelPercentage (float): Percentage of white pixels found in img.
    """ 

    
    img= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret,img = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
    Count = cv2.countNonZero(img)

    dimensions = img.shape
    total_pixels = dimensions[0]*dimensions[1]

    factor = total_pixels #*(nozzle_diameter_pixels/100)
    return Count/factor*100


def analyze_blur(vid,white_threshold=75,moving_treshold=75,crop_x = 450,crop_y = 450,mode="Full"):
    """Analyzes the blur of a video. Subtracts the first frame from the 'blurred images' and 
    accumulates the result into a single image, this is then thresholded. 

    Args:
        vid (_type_): video object from utils vid class.
        show_frame (bool, optional): ipython plot of first frame and accumulated image. Defaults to False.
        white_threshold (int, optional): threshold value. Defaults to 75.
        console_output (bool, optional): prints image movement range and blurriness percentage. Defaults to False.

    Returns:
        Percentage (float): Blurriness percentage
    """    

    frame_id1 = vid.get_frameid(0, 0)
    first_moving_frame = 1 #get_moving_frame(vid,moving_treshold)
    last_moving_frame = vid.get_duration() #returns the last frame_id

    frame1 = vid.get_frame(frame_id1)
    
    #Show the initial frame
    if vid.debug.blur: vid.show_frame(frame1)

    height, width, channels = frame1.shape
    # Create an empty image to accumulate the white pixels
    white_pixels_accumulated = np.zeros((crop_x, crop_y, channels), np.uint8)

    if mode == "Full":
        image_range = range(first_moving_frame, last_moving_frame)
    elif mode == "LinearOnly":
        offset = 0.25
        before_jerk = int((last_moving_frame/2)-offset*last_moving_frame)
        after_jerk = int((last_moving_frame/2)+offset*last_moving_frame)
        image_range = list(range(first_moving_frame, before_jerk)) + list(range(after_jerk,last_moving_frame))
    else:
        warnings.warn("mode should be Full or LinearOnly not ", mode," So mode is set too Full")
        image_range = range(first_moving_frame, last_moving_frame)

    crop = utils.transform(crop_x,crop_y,0,0)
    frame1 = vid.crop_to_roi(frame1,transform=crop)
    
    for img_id in image_range:
        frame2 = vid.get_frame(img_id)
        frame2 = vid.crop_to_roi(frame2,transform=crop)
        img = cv2.subtract(frame2,frame1)
        #TODO subtract from more than 1 image to ensure blurring instead of lighting?
        #Convert to Gray values
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to create a binary image

        _, thresh = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)

        # Create a mask for the white pixels
        white_pixels_mask = cv2.merge([thresh, thresh, thresh])

        # Accumulate the white pixels
        white_pixels_accumulated = cv2.add(white_pixels_accumulated, white_pixels_mask)

    nozzle_diameter_pixels = vid.find_nozzle_diameter_pixels(search_offset=3,redudancy=3)
    percentage = count_white(white_pixels_accumulated,nozzle_diameter_pixels)
    
    #Show the accumulated white pixel frame
    if vid.debug.blur: vid.show_frame(white_pixels_accumulated)
    #Print the range of moving images.
    if vid.debug.CLI: print("Movement in "+str(image_range))

    return float(percentage),float(nozzle_diameter_pixels),white_pixels_accumulated

def blur_analysis_series(amount_of_runs,
            classes,
            variable,
            file_extension,
            variable_start,
            variable_end,
            variable_step,
            subfolder,
            vibration_threshold,
            debug = utils.debug(),
            crop_x=450,
            crop_y=450,
            mode = "Full"         
            ):
    """blur analysis for a series of videos

    Args:
        amount_of_runs (int): N of the experiment per class
        classes (list): experiment classes
        variable (str): the variable in the video name
        file_extension (str): extension of the video file
        variable_start (int): start value for the range
        variable_end (int): end value for the range
        variable_step (int): stepsize for the range
        subfolder (str): folder name containing videos
        vibration_threshold (int): _description_
        console_output (bool): toggles print statements to console

    Returns:
        df (pd.DataFrame()): Panda DataFrame containing classes, 
        speed and corresponding blurriness percentage
    """            
    df_classes=[]
    df_speed=[]
    df_blur=[]
    df_run=[]
    df_nozzle=[]
    for prefix in classes:

        for run in range(1,(int(amount_of_runs)+1)):
            filenames = []
            speed_list = []
            run_name = "Run"+str(run)
        
            for i in range(variable_start,(variable_end+variable_step),variable_step):
                filenames.append(variable+str(i)+file_extension)
                speed_list.append(i)

            for i in range(len(filenames)):
                filename = filenames[i]
                speed = speed_list[i]
                __location__ = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
                
                filepath = __location__ +'\\'+ subfolder +'\\' + prefix+ str(run_name)+'\\' + filename

                if os.path.exists(filepath):
                    video = utils.vid(filepath,debug)

                    percentage,nozzle_diameter_pixels,white_pixels_accumulated = analyze_blur(video,white_threshold=vibration_threshold,crop_x=crop_x,crop_y=crop_y,mode = mode)
                    
                    if video.debug.CLI: print(str(filename)+' has '+str(round(percentage,2))+"%"+' blur \n')
                    #Save frame
                    frame_count = str(speed)
                    path = 'frames_output\\'+prefix+'\\' +str(run_name)
                    master_folder = 'frames_output\\'+prefix
                    output_location = 'frames_output\\'

                    if not os.path.exists(output_location): 
                        os.mkdir(output_location) 

                    if not os.path.exists(master_folder): 
                        os.mkdir(master_folder) 

                    if not os.path.exists(path): 
                        os.mkdir(path)

                    output_name = path+"\\frame%s.jpg" % frame_count
                    cv2.imwrite(output_name, white_pixels_accumulated)  

                    if prefix == 'Og': df_classes.append('Negative Control')
                    elif prefix == 'SLS': df_classes.append('Nylon 12')
                    else: df_classes.append(prefix)
                    df_speed.append(int(speed))
                    df_blur.append(percentage)
                    df_run.append(run)
                    df_nozzle.append(nozzle_diameter_pixels)
                else:
                    warnings.warn("Could not find video at "+str(filepath))

    df = pd.DataFrame()
    df['Classes'] = df_classes
    df['run'] = df_run
    df['Speed'] = df_speed
    df['Blur'] = df_blur
    df['Nozzle Diameter in Pixels'] = df_nozzle
    
    return df
        