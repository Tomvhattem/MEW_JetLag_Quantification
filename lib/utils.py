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
import seaborn as sns
import statistics

class transform:
    def __init__(self,width,height,x_shift,y_shift):
        """transform object used for crop and shift

        Args:
            width (int): width used for x crop, with respect to the centre
            height (int): height used for y crop, with respect to the centre
            x_shift (int): shift in x direction with respect to the centre
            y_shift (int): shift in y direction with respect to the centre
        """        
        self.width = width
        self.height = height
        self.x_shift = x_shift
        self.y_shift = y_shift
        
class debug:
    def __init__(self,
                segmentation=False,
                scharr_segmentation=False,
                calibration=False,
                rotation=False,
                search_height = False,
                height_threshold = False,
                annotate=False,
                threshold = False,
                CLI=False,
                error=False,
                jetlag = False,
                blur=False,
                CP=False):
        self.segmentation = segmentation
        self.scharr_segmentation = scharr_segmentation
        self.calibration = calibration
        self.rotation = rotation
        self.search_height = search_height
        self.height_threshold = height_threshold
        self.threshold = threshold
        self.annotate = annotate
        self.CLI = CLI
        self.error = error
        self.jetlag = jetlag
        self.blur = blur
        self.CP=CP


class vid:
    def __init__(self,filepath,debug, calibration_frame='Start'):
        """Basic functions for analyzing openCV2 Videos,
        requires the filepath to a video file"""
        self.debug = debug
        if not os.path.exists(filepath): warnings.warn("Path Does Not Exist", filepath)
        self.video =cv2.VideoCapture(filepath)
        self.current_frame = self.get_frame(2)
        self.height,self.width = self.get_resolution()
        self.transform = transform(self.width,self.height,0,0)
        self.total_frames = self.get_duration()
        self.set_calibration_frame(calibration_frame) #Default frame for calibration is the first frame        
        self.path = filepath
    
        
    def set_error_log(self,boolean=True):
        self.error_log = boolean
        
    def set_calibration_frame(self,frame_location):
        if frame_location == 'Start':
            if self.debug.CLI: 
                print('frame used for calibration is 1')
            self.calibration_frame = 1
            
        elif frame_location == 'End':
            if self.debug.CLI: 
                print('frame used for calibration is ',self.total_frames-1)
            self.calibration_frame = int(self.total_frames-1)

        elif frame_location == 'Centre':
            if self.debug.CLI: 
                print('frame used for calibration is ',(self.total_frames +1)/2)
            self.calibration_frame = int((self.total_frames +1)/2)
        else:
            warnings.warn('Please use Start, End or Centre strings to indicate the frame location. Defaults to 1')
            self.calibration_frame = 1
        
    def get_frameid(self,minute,sec):
        """Given a time in minute and seconds returns the frame id
        taking the frame rate into account."""
        fps = self.video.get(cv2.CAP_PROP_FPS)
        return int(fps*(minute*60 + sec))

    def get_duration(self):
        """Returns the last frame of the video"""
        fps = self.video.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        return frame_count

    def get_frame(self,frame_id):
        """Returns image/frame from the video for a given frame_id."""
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.video.read()
        return frame
    
    def get_current_frame(self):
        """returns the frame set by set_current_frame()

        Returns:
            current_frame (ndarray): current frame
        """        
        return self.current_frame

    def set_current_frame(self,frame_id):
        """initialized the current frame by frame number. Use get_frameid() for frameid by time

        Args:
            frame_id (int): frame id, independent of fps/time
        """        
        self.current_frame = self.get_frame(frame_id)

    def show_frame(self,frame):
        """Uses ipython jupiter library and matplotlib to show the image"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
    
    def get_resolution(self):
        """returns the x,y resolution of the currently set frame

        Returns:
            height(int): resolution height of current frame
            width(int): resolution width of current frame
        """        
        height,width  = self.current_frame.shape[:2]
        return height,width
    
    def show_current_frame(self):
        """ visualize the current frame (set by set_current_frame())
        """        
        self.show_frame(self.current_frame)

    def set_transform(self,transform):
        """sets the transform object created by the transform class for the current object.
        Used for cropping and shifting the image. 
        Args:
            transform (object): transform object
        """        
        self.transform = transform

    def crop_to_roi(self,frame,transform=None):
        """Using the transform object initialized by the set_transform() method, crop and shift
        the image. The first part of the transform object contains the width and heigth, the 3rd and 4th
        contain the x and y shift.  

        Args:
            frame (Image Array): OpenCV Image Frame
            x_width (int): Crop Width with respect to centre
            y_height (int): Crop Heigth with respect to centre
            x_shift (int, optional): Shift in x direction. (left is negative, right is positive) Defaults to 0.
            y_shift (int, optional): shift in y direction (lower is negative, higher positive). Defaults to 0.

        Returns:
            Cropped Image (ndarray): OpenCV Image 
        """    
        if transform is None:
            transform = self.transform
            
        x_start = int((self.width/2 + transform.x_shift) - transform.width/2)
        x_end = int((self.width/2 + transform.x_shift) + transform.width/2)
        y_start = int((self.height/2 + transform.y_shift) - transform.height/2)
        y_end = int((self.height/2 + transform.y_shift) + transform.height/2)

        cropped_frame = frame[y_start:y_end,x_start:x_end]
        return cropped_frame

    def crop_current_frame(self):
        """ Uses the crop_to_roi() function on the current frame, set by set_current_frame()

        Returns:
            cropped_frame (ndarray): Return cropped image
        """        
        cropped_frame = self.crop_to_roi(self.current_frame)
        return cropped_frame

    def plot_histogram(self,frame):
        """Plot the histogram of the given frame

        Args:
            frame (ndarray): image
        """        
        hist = cv2.calcHist([frame],[0],None,[256],[0,256])
        plt.plot(hist)
        plt.show()
    
    def convert_grayscale(self,frame):
        """Convert image from BGR to Grayscale format

        Args:
            frame (ndarray): BGR image (not RGB)

        Returns:
            frame (ndarray): grayscale image
        """        
        return cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    def convert_color(self,frame):
        """Convert grayscale image to RGB

        Args:
            frame (ndarray): grayscale image

        Returns:
            frame (ndarray): RGB image
        """        
        return cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)

    def convert_current_grayscale(self):
        """Convert Current frame to grayscale
        """        
        try: self.current_frame = self.convert_grayscale(self.current_frame)
        except: warnings.warn("Can't convert current frame to grayscale")

    def find_y_edge(self,frame,x,y,direction='Up',minimum=0,maximum=None):
        """ find white pixel on a y direction.

        Args:
            frame (ndarray): Black/white (segmented) grayscale image
            x (int): start location on the x axis
            y (int): start location on the y axis
            direction (str, optional): search direction, up or down. Defaults to 'Up'.
            minimum (int, optional): minimum y value in image when searching down. Defaults to 0.
            maximum (_type_, optional): maximum y value in image when searching up. Defaults to self.height-1.

        Returns:
            y (int): y value where 'white' pixel (value above 230) is found
        """        
        if maximum==None: maximum = self.height-1
        if direction == 'Up': factor = -1
        elif direction == 'Down': factor = 1
        else: raise ValueError("Value should be Up or Down ",direction, ' does not exist')

        while (y > minimum) and (y<maximum):
            if frame[int(y), int(x)] > 230:  # Check if the pixel is white
                break
            y = y+ factor*1
        return y

    def find_y_edge_black(self,frame,x,y,direction='Up',minimum=0,maximum=None):
        """ find black pixel on a y direction.

        Args:
            frame (ndarray): Black/white (segmented) grayscale image
            x (int): start location on the x axis
            y (int): start location on the y axis
            direction (str, optional): search direction, up or down. Defaults to 'Up'.
            minimum (int, optional): minimum y value in image when searching down. Defaults to 0.
            maximum (_type_, optional): maximum y value in image when searching up. Defaults to self.height-1.

        Returns:
            y (int): y value where 'white' pixel (value above 230) is found
        """        
        if maximum==None: maximum = self.height-1
        if direction == 'Up': factor = -1
        elif direction == 'Down': factor = 1
        else: raise ValueError("Value should be Up or Down ",direction, ' does not exist')

        while (y > minimum) and (y<maximum):
            if frame[int(y), int(x)] < 20 :  # Check if the pixel is Black
                break
            y = y+ factor*1
        return y

    def find_x_edge(self,frame,x,y,direction='Right',minimum=0,maximum=None):
        """ find white pixel on a x direction.

        Args:
            frame (ndarray): Black/white (segmented) grayscale image
            x (int): start location on the x axis
            y (int): start location on the y axis
            direction (str, optional): search direction, Left or Right. Defaults to 'Right'.
            minimum (int, optional): minimum y value in image when searching down. Defaults to 0.
            maximum (_type_, optional): maximum y value in image when searching up. Defaults to self.height-1.

        Returns:
            y (int): y value where 'white' pixel (value above 230) is found
        """  
        if maximum==None: maximum = self.width-1
        if direction == 'Left': factor = -1
        elif direction == 'Right': factor = 1
        else: raise ValueError("Value should be Left or Right ",direction, ' does not exist')
        
        while (x > minimum) and (x< maximum):
            if frame[int(y), int(x)] > 230:  # Check if the pixel is white
                break
            x = x + factor*1
        return x
    
    def blur_canny_edge(self,frame,blur=17,canny_lower=10,canny_upper=15):
        """Converts image to grayscale if it is not already. Applies a gaussian
        blur filter on frame, with kernel of size blur, then applies a canny edge filter, with lower and upper bounds. 
        Can show all 3 intermediate results

        Args:
            frame (ndarray): frame to be filtered
            blur (int, optional): value for gaussian blur kernel. Defaults to 17.
            canny_lower (int, optional): lower limit of the canny edge filter. Defaults to 10.
            canny_upper (int, optional): upper limit of the canny edge filter. Defaults to 15.
            deubg (bool, optional): plot 3 intermediate processing frames. Defaults to False.

        Returns:
            canny_edge_image (ndarray): filtered image. 
        """        
        try: frame = self.convert_grayscale(frame)
        except: print('frame is not BGR but already Grayscale')
        if self.debug.segmentation: 
            print('canny edge frame')
            self.show_frame(frame)

        blurred = cv2.GaussianBlur(frame, (blur,blur), 0)
        if self.debug.segmentation: 
            print('Blurr from Canny Edge')
            self.show_frame(blurred)

        edges = cv2.Canny(blurred, canny_lower,canny_upper) 
        if self.debug.segmentation: 
            print('Edges from canny edge')
            self.show_frame(edges)

        return edges

    def scharr_top_segmentation(self,frame):
        if self.debug.scharr_segmentation: 
            print('Original Frame scharr top segmentation')
            self.show_frame(frame)

        frame_gray = self.convert_grayscale(frame)
        img = cv2.GaussianBlur(frame_gray,(15,15),0)
        gradient_x = cv2.Scharr(img, cv2.CV_64F, 0, 1)  
        if self.debug.scharr_segmentation:
            print('gradient in x direction, scharr_top_segmentation')
            self.show_frame(gradient_x.astype(np.uint8))

        abs_gradient_x = cv2.convertScaleAbs(gradient_x)
        _, thresholded = cv2.threshold(abs_gradient_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.debug.scharr_segmentation: 
            print('thresholded image from scharr_top_segmentation')
            self.show_frame(thresholded)

        kernel = np.ones((5, 5), np.uint8) #kernel used for closing operation
        closed_scharr = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        if self.debug.scharr_segmentation: 
            print('closed scharr top segmentation')
            self.show_frame(closed_scharr)
        return closed_scharr
    

    def scharr_nozzle_segmentation(self,frame):
        #Unused
        if self.debug.scharr_segmentation: 
            print('Original Frame scharr nozzle segmentation')
            self.show_frame(frame)

        frame_gray = self.convert_grayscale(frame)
        img = cv2.GaussianBlur(frame_gray,(15,15),0)
        gradient_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)  
        if self.debug.scharr_segmentation:
            print('gradient in x direction, scharr_nozzle_segmentation')
            self.show_frame(gradient_x.astype(np.uint8))

        abs_gradient_x = cv2.convertScaleAbs(gradient_x)
        _, thresholded = cv2.threshold(abs_gradient_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.debug.scharr_segmentation: 
            print('thresholded image from scharr_nozzle_segmentation')
            self.show_frame(thresholded)

        kernel = np.ones((5, 5), np.uint8) #kernel used for closing operation
        closed_scharr = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        if self.debug.scharr_segmentation: 
            print('closed scharr nozzle segmentation')
            self.show_frame(closed_scharr)
        return closed_scharr

    def correct_rotation_legacy(self,frame,delineation_size=5):
        edges = self.blur_canny_edge(frame)

        #Delineation. 
        kernel = np.ones((delineation_size, delineation_size), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        if debug.rotation: 
            print('rotation legacy dilated edges')
            self.show_frame(dilated_edges)

        
        left = self.find_y_edge(dilated_edges,1,int(2*self.height/3),direction='Up') #find the first white pixel on the left
        right = self.find_y_edge(dilated_edges,(self.width-1),int(2*self.height/3),direction='Up') #find the first white pixel on the right

        Opposite = (left-right)
        Adjacent = self.width
        Angle = math.tan(Opposite/Adjacent)
        Angle_degrees = np.rad2deg(Angle)
        self.angle = Angle_degrees
        self.angle_rad = Angle
        self.set_rotation_matrix()

        rotation_matrix = cv2.getRotationMatrix2D((self.width / 2, self.height / 2), -Angle_degrees, 1)
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (self.width, self.height))

        if self.debug.rotation: 
            print('rotation legacy rotated_frame')
            self.show_frame(rotated_frame)

        return rotated_frame

    def correct_rotation(self,frame,delineation_size=5):
        segmented_image = self.scharr_top_segmentation(frame)
        
        dilated_edges = segmented_image

        if self.debug.rotation: 
            print('Threshold Image from Correct Rotation')
            self.show_frame(dilated_edges)
        redudancy = 10
        total_left = []
        total_right = []
        offset_step = 10
        for i in range(redudancy):
            x_offset = i*offset_step
            left_local = self.find_y_edge(dilated_edges,1 + x_offset ,int(2*self.height/3),direction='Up') #find the first white pixel on the left
            right_local = self.find_y_edge(dilated_edges,(self.width-1)-x_offset,int(2*self.height/3),direction='Up') #find the first white pixel on the right
            total_left.append(left_local)
            total_right.append(right_local)

        left = np.median(total_left)
        right = np.median(total_right)


        Opposite = (left-right)
        Adjacent = self.width
        Angle = math.tan(Opposite/Adjacent)
        Angle_degrees = np.rad2deg(Angle)
        self.angle = Angle_degrees
        self.angle_rad = Angle
        self.set_rotation_matrix()

        rotation_matrix = cv2.getRotationMatrix2D((self.width / 2, self.height / 2), -Angle_degrees, 1)
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (self.width, self.height))

        if self.debug.rotation: 
            print('show_rotated_frame')
            self.show_frame(rotated_frame)
        return rotated_frame
    
    def rotate_frame(self,frame,angle=None):
        if angle==None: angle=self.angle
        rotation_matrix = cv2.getRotationMatrix2D((self.width / 2, self.height / 2), -angle, 1)
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (self.width, self.height))
        return rotated_frame

    def rotate_current_frame(self):
        self.current_frame = self.rotate_frame(self.current_frame,self.angle)

    def height_threshold(self,frame_gray):
        """Main segmentation, also used in frameanalysis although that version is a tweaked duplicate."""
        #First Threshold
        ret,threshold = cv2.threshold(frame_gray,70,255,cv2.THRESH_BINARY)
        if self.debug.height_threshold: 
            print('height_threshold')
            self.show_frame(threshold)

        #Adaptive Threshold for Masking
        ada_threshold_image = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 9)
        if self.debug.height_threshold: 
            print('adaptive threshold')
            self.show_frame(ada_threshold_image)

        kernel = np.ones((2, 2), np.uint8) #kernel used for closing operation
        closed_image = cv2.morphologyEx(ada_threshold_image, cv2.MORPH_CLOSE, kernel)
        closed_image = 255 - closed_image #Invert Image
        if self.debug.height_threshold: 
            print('Closed Image - height_threshold ')
            self.show_frame(closed_image)

        kernel = np.ones((12,12), np.uint8)
        dilated_edges = cv2.dilate(closed_image, kernel, iterations=1)
        if self.debug.height_threshold: 
            print('Dilated Image - height_threshold')
            self.show_frame(dilated_edges)
        combined_image = cv2.bitwise_and(threshold, dilated_edges)

        return combined_image

    def find_search_height(self,delineation_size=5,y_offset=30,mode=""):
        frame = self.get_frame(self.calibration_frame)

        self.corrected_frame = self.correct_rotation(frame,delineation_size=delineation_size)
        if self.debug.search_height and mode != 'nozzle_bottom': 
            print('rotation corrected frame')
            self.show_frame(self.corrected_frame)

        frame = self.rotate_frame(frame)
        frame_gray = self.convert_grayscale(frame)
        
        combined_image = self.height_threshold(frame_gray)

        segmented_image = self.scharr_top_segmentation(frame)

        if self.debug.search_height and mode != 'nozzle_bottom':
            print('search_height segementation:')
            self.show_frame(segmented_image)

        x_offset_step = 10
        redudancy = 5
        search_height = []
        for i in range(redudancy):
            x_offset = x_offset_step + i*x_offset_step
            
            search_height.append(self.find_y_edge(segmented_image,int(x_offset),int(2*self.height/3)))
        
        search_height_final = np.median(search_height) + y_offset
        return search_height_final,combined_image

    def find_nozzle_diameter_pixels_legacy(self,delineation_size = 5,return_location=False,redudancy = 3,temp_offset = 1,search_offset = 20,mode=''):
        search_height,dilated_edges = self.find_search_height(delineation_size=delineation_size,y_offset=search_offset)
        
        if self.debug.calibration and mode != 'nozzle_bottom': 
            print('nozzle diameter dilated edges: ')
            self.show_frame(dilated_edges)
            
        self.search_height = search_height
        left_edge_nozzle_list = []
        right_edge_nozzle_list = []
        for i in range(redudancy):
            search = search_height+temp_offset*i
            left_edge_nozzle_list.append( self.find_x_edge(dilated_edges,20,search,direction='Right'))
            right_edge_nozzle_list.append(self.find_x_edge(dilated_edges,self.width-20,search,direction='Left'))

        left_edge_nozzle = statistics.median(left_edge_nozzle_list)
        right_edge_nozzle = statistics.median(right_edge_nozzle_list)

        nozzle_diameter_pixels = right_edge_nozzle - left_edge_nozzle
        show_error = False
        while (nozzle_diameter_pixels > 60) and ((search_height+temp_offset*3)<self.height):
            left_edge_nozzle_list = []
            right_edge_nozzle_list = []
            for i in range(redudancy):
                search = search_height+temp_offset*i
                left_edge_nozzle_list.append( self.find_x_edge(dilated_edges,20,search,direction='Right'))
                right_edge_nozzle_list.append(self.find_x_edge(dilated_edges,self.width-20,search,direction='Left'))

            left_edge_nozzle = statistics.median(left_edge_nozzle_list)
            right_edge_nozzle = statistics.median(right_edge_nozzle_list)
            nozzle_diameter_pixels = right_edge_nozzle - left_edge_nozzle

            search_height += 2
            
            show_error=True

        if show_error: 
            warnings.warn("Found abnormal nozzle diameter, possibly due to noise, searching lower height. Inspect image if hair or dust is visible and if it is possible to find the correct diameter at lower height. Image at path: "+str(self.path))
            if self.debug.error:
                print('Show Dilated Edges for abnormal nozzle diameter')
                self.show_frame(dilated_edges)

        
        self.final_search_offset = search_height+temp_offset*i
        if return_location: return nozzle_diameter_pixels,left_edge_nozzle,right_edge_nozzle,search_height
        
        return nozzle_diameter_pixels
    

    def find_nozzle_diameter_pixels(self,delineation_size = 5,return_location=False,redudancy = 10,temp_offset = 1,search_offset = 10,mode=''):
        #! Changed redudancy to 10 from 3 
        search_height,dilated_edges_og = self.find_search_height(delineation_size=delineation_size,y_offset=search_offset,mode=mode)

        #Closing on dilated edges to get better consistancy
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)) 
        #! WAS 5 instead of 7
        closed_image = cv2.morphologyEx(dilated_edges_og, cv2.MORPH_CLOSE, closing_kernel)

        dilated_edges = closed_image

        self.search_height = search_height
        left_edge_nozzle_list = []
        right_edge_nozzle_list = []

        start_x_left = int(self.width * 0.30) #was 20
        start_x_right = int(self.width * 0.70) #was self.width - 20
        k=0
        for i in range(redudancy):
            search = search_height+temp_offset*i
            left_edge_nozzle_list.append( self.find_x_edge(dilated_edges,start_x_left,search,direction='Right'))
            right_edge_nozzle_list.append(self.find_x_edge(dilated_edges,start_x_right,search,direction='Left'))
            k=i
        left_edge_nozzle = statistics.median(left_edge_nozzle_list)
        right_edge_nozzle = statistics.median(right_edge_nozzle_list)

        nozzle_diameter_pixels = right_edge_nozzle - left_edge_nozzle
        show_error = False
        while (nozzle_diameter_pixels > self.width*0.046) and ((search_height+temp_offset*3)<self.height):
            left_edge_nozzle_list = []
            right_edge_nozzle_list = []
            for i in range(redudancy):
                search = search_height+temp_offset*i
                left_edge_nozzle_list.append( self.find_x_edge(dilated_edges,20,search,direction='Right'))
                right_edge_nozzle_list.append(self.find_x_edge(dilated_edges,self.width-20,search,direction='Left'))
                k=i
            left_edge_nozzle = statistics.median(left_edge_nozzle_list)
            right_edge_nozzle = statistics.median(right_edge_nozzle_list)
            nozzle_diameter_pixels = right_edge_nozzle - left_edge_nozzle

            search_height += 1
            show_error=True

        if show_error: 
            warnings.warn("Found abnormal nozzle diameter, possibly due to noise, searching lower height. Inspect image if hair or dust is visible and if it is possible to find the correct diameter at lower height. Image at path: "+str(self.path))
            if self.debug.error:
                print('Show Dilated Edges for abnormal nozzle diameter')
                self.show_frame(dilated_edges)

        if self.debug.calibration and mode != 'nozzle_bottom': 
            print('nozzle diameter dilated edges: ')
            self.show_frame(dilated_edges)
            print(self.path)
            print("search height",search_height)
            print("Nozzle Diameter Pixels",nozzle_diameter_pixels)

        self.final_search_offset = search_height+temp_offset*i
        if return_location: return nozzle_diameter_pixels,left_edge_nozzle,right_edge_nozzle,search_height
        
        return nozzle_diameter_pixels

    def nozzle_calibration(self,nozzle_diameter=0.5):
        nozzle_diameter_pixels,left_edge_nozzle,right_edge_nozzle,search_height= self.find_nozzle_diameter_pixels(return_location=True)
        mm_per_pixel = nozzle_diameter / nozzle_diameter_pixels 
        return mm_per_pixel,nozzle_diameter_pixels,left_edge_nozzle,right_edge_nozzle,search_height

    def set_rotation_matrix(self):
        angle = -self.angle_rad
        self.origin = ((self.width/2),(self.height/2))
        self.R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])

    def rotate_coordinates(self,p):
        """Rotates the coordinates from initialisation rotation back into original system"""
        o = np.atleast_2d(self.origin)
        p = np.atleast_2d(p)
        coordinates = np.squeeze((self.R @ (p.T-o.T) + o.T).T)
        return (int(coordinates[0]), int(coordinates[1]))

    def add_circle(self,frame,coordinates,color=(0, 0, 255)):
        # Draw a red circle at the location
        radius = 3  # Radius of the circle
        thickness = 10  # Thickness of the circle's outline

        coordinates = self.rotate_coordinates(coordinates)
        return cv2.circle(frame, coordinates, radius, color, thickness)





class Jet_Lag:
    def __init__(self,filepath,local_debug=None,nozzle_diameter=0.5,calibration_frame='Start',CP_mode='Legacy',CP_calibrated_distance=3,main_threshold=70):
        if local_debug == None: local_debug = debug()
        self.video = vid(filepath,local_debug,calibration_frame=calibration_frame)

        mm_per_pixel, nozzle_diameter_pixels,left_edge_nozzle,right_edge_nozzle,search_height = self.video.nozzle_calibration(nozzle_diameter = nozzle_diameter)
        
        self.mm_per_pixel = mm_per_pixel
        self.nozzle_diameter_pixels = nozzle_diameter_pixels
        self.left_edge_nozzle = left_edge_nozzle
        self.right_edge_nozzle = right_edge_nozzle
        self.x_centre_nozzle = (left_edge_nozzle + right_edge_nozzle) /2 
        self.search_height = search_height
        self.nozzle_centre = (int(self.x_centre_nozzle),int(self.search_height))
        self.main_threshold = main_threshold

        tmp = 'frames_output\\jet_lag_tmp\\'
        if not os.path.exists(tmp): os.makedirs(tmp)
        self.set_output_folder(tmp)
        
        self.find_nozzle_bottom()

        if CP_mode == 'Legacy':
            frame2 = self.video.get_frame(self.video.calibration_frame)
            segmented_image2 = self.CP_segmentation(frame2) #Segmentation of the image

            self.find_CP(segmented_image2) #Collector Plate Height
            self.distance_nozzle_CP_pixels = abs(self.bottom_nozzle_y - self.CP_height)
            self.distance_nozzle_CP_mm = self.distance_nozzle_CP_pixels * self.mm_per_pixel

        elif CP_mode == 'Calibration':
            self.find_CP_by_calibration(CP_calibrated_distance)
        else:
            raise ValueError('CP_mode should be Legacy or Calibration not', CP_mode)

        if self.video.debug.CLI:
            print('Nozzle Diameter in Pixels: ',self.nozzle_diameter_pixels)
            print('X_centre nozzle: ',self.x_centre_nozzle)
            print('search height: ',self.search_height)
            if CP_mode == 'Legacy':
                print('Distance Nozzle Collector Plate: ',self.distance_nozzle_CP_mm,' [mm]')
        

    def find_nozzle_bottom(self,offset = 1,margin = 2):
        #1 setting margin to 5 is a lot more consistant. Or 2.
        redudancy = 1
        smaller_nozzle_diameter_pixels = self.video.find_nozzle_diameter_pixels(search_offset=offset,redudancy=redudancy,mode='nozzle_bottom')
        
        while smaller_nozzle_diameter_pixels >= (self.nozzle_diameter_pixels-margin):
            offset+=1
            smaller_nozzle_diameter_pixels = self.video.find_nozzle_diameter_pixels(search_offset=offset,redudancy=redudancy,mode='nozzle_bottom')

        self.bottom_nozzle_y = self.search_height + offset
        self.bottom_nozzle = (self.x_centre_nozzle,self.bottom_nozzle_y)


    def segmentation(self,frame):
        """"Main segmentation on a frame. threshold, adaptive threshold, closing on adaptive threshold"""
        frame_gray = self.video.convert_grayscale(frame)

        #First Threshold
        ret,threshold = cv2.threshold(frame_gray,self.main_threshold,255,cv2.THRESH_BINARY)

        #Adaptive Threshold for Masking
        ada_threshold_image = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 9)

        kernel = np.ones((2, 2), np.uint8) #kernel used for closing operation
        closed_image = cv2.morphologyEx(ada_threshold_image, cv2.MORPH_CLOSE, kernel)
        closed_image = 255 - closed_image #Invert Image

        kernel = np.ones((12,12), np.uint8)
        dilated_edges = cv2.dilate(closed_image, kernel, iterations=1)

        combined_image = cv2.bitwise_and(threshold, dilated_edges)

        #Closing on dilated edges to get better consistancy
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)) #! was 5
        closed_image = cv2.morphologyEx(combined_image, cv2.MORPH_CLOSE, closing_kernel)

        return closed_image
    
    def CP_segmentation(self,frame):
        frame_gray = self.video.convert_grayscale(frame)

        #First Threshold
        ret,threshold = cv2.threshold(frame_gray,70,255,cv2.THRESH_BINARY)

        #Adaptive Threshold for Masking
        ada_threshold_image = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 9)

        kernel = np.ones((2, 2), np.uint8) #kernel used for closing operation
        closed_image = cv2.morphologyEx(ada_threshold_image, cv2.MORPH_CLOSE, kernel)
        closed_image = 255 - closed_image #Invert Image

        kernel = np.ones((12,12), np.uint8)
        dilated_edges = cv2.dilate(closed_image, kernel, iterations=1)

        combined_image = cv2.bitwise_and(threshold, dilated_edges)
        return combined_image

    def find_CP(self,segmented_image,redundancy = 20):
        y_top= int(self.video.height*0.65) #search start location

        min_cp_height, max_cp_height = self.find_init_direction(segmented_image) #determine left or right search
        edge_offset = 100
        step = 25
        centre_margin = 150
        safety_offset = 0

        if self.init_direction == 'Left':
            start_x = edge_offset
            offset_step = step

        if self.init_direction == 'Right':
            start_x = self.video.width - edge_offset
            offset_step = -step


        if self.video.debug.CP: 
            print('segmented image from find CP')
            self.video.show_frame(segmented_image)
        CP_heights = []
        CP_thickness = []
        for i in range(redundancy):
            x = start_x + offset_step*i
            
            if self.init_direction == 'Left':
                if x > (self.video.width / 2 - centre_margin):
                    break
            elif self.init_direction =='Right':
                if x < (self.video.width / 2 + centre_margin):
                    break

            #finds the top of the already printed line to find the first white pixel
            top = self.video.find_y_edge(segmented_image,x,y_top,direction='Down')
            #Starts at the previously found location until there is a black pixel
            bottom = self.video.find_y_edge_black(segmented_image,start_x,top,direction='Down')
            thickness = bottom - top #index at bottom is higher
            
            local_CP_height = (top+bottom)/2

            if not (local_CP_height > self.video.height*0.9):
                CP_heights.append(local_CP_height)
                CP_thickness.append(thickness)
            else:
                #print("collector plate height: ",local_CP_height,"Thickness ",thickness)
                #print(CP_heights,CP_thickness)
                warnings.warn("Something went wrong with the collector plate height at the first location trying again")
                

        smallest = min(CP_thickness)
        index = CP_thickness.index(smallest)
        self.CP_height = CP_heights[index]

        max_diameter_fibre = 10 #after thresholding so on the large side to be sure
        if smallest > max_diameter_fibre:
            if abs(max_cp_height - min_cp_height) < max_diameter_fibre:
                print('minimum height + difference in height')
                self.CP_height = min_cp_height + abs(max_cp_height - min_cp_height)

            else:
                print('issues with CP height')
                self.CP_height = min_cp_height
        
        #self.CP_height = np.median(CP_heights)
        self.CP_height = self.CP_height + safety_offset
        self.NP = (self.x_centre_nozzle,self.CP_height)


    def find_CP_by_calibration(self,CP_calibrated_distance=3):
        distance_yellow_green = CP_calibrated_distance * (1/self.mm_per_pixel)

        self.CP_height = self.bottom_nozzle_y + distance_yellow_green
        self.NP = (self.x_centre_nozzle,self.CP_height)

    def find_JCP(self,segmented_image):
        self.JCP_x = self.video.find_x_edge(segmented_image,int(self.x_centre_nozzle),int(self.CP_height),direction=self.direction)
        self.JCP = (self.JCP_x,self.CP_height)

    def calculate_jet_lag(self):
        self.JetLag = abs(self.JCP_x - self.x_centre_nozzle)*self.mm_per_pixel

    def annotate_image(self,segmented_image,mode='show'):
        combined_image_color = self.video.convert_color(segmented_image)
        #self.video.show_frame(segmented_image)
        marked_image = self.video.add_circle(combined_image_color,self.nozzle_centre,(255,0,0))
        marked_image = self.video.add_circle(marked_image,self.NP,(0,255,0))

        if abs(self.JetLag) < 7:
            marked_image = self.video.add_circle(marked_image,self.JCP,(0,0,255))
        marked_image = self.video.add_circle(marked_image,self.bottom_nozzle,(0,255,255))
        if self.video.debug.annotate and (mode == 'show'): 
            print('Annotated Image')
            self.video.show_frame(marked_image)
        return marked_image

    def find_jet_direction(self,segmented_image):
        #Exactly at the centre
        #centre_nozzle_CP = (int(self.CP_height) + int(self.search_height))/2 

        #Where to start search for the jet at the top to find the direction
        #Too high fails at low speeds, too low fails with dust on CP
        centre_nozzle_CP = int(self.search_height) + 0.5* abs(int(self.CP_height) - int(self.search_height))
        
        #How many pixels left and right it searches, minimize time per frame for real time.
        search_area = 80

        minimum = self.x_centre_nozzle - search_area
        maximum = self.x_centre_nozzle + search_area

        value = self.video.find_x_edge(segmented_image,
                        int(self.x_centre_nozzle),
                        int(centre_nozzle_CP),
                        direction='Left',
                        minimum=minimum,
                        maximum=maximum)

        value2 = self.video.find_x_edge(segmented_image,
                        int(self.x_centre_nozzle),
                        int(centre_nozzle_CP),
                        direction='Right',
                        minimum=minimum,
                        maximum=maximum)

        left = abs(value - int(self.x_centre_nozzle))
        right = abs(value2 - int(self.x_centre_nozzle))
        self.direction = 'Left'
        if right<left:
            self.direction = 'Right'

    def find_init_direction(self,segmented_image,redudancy = 3,border = 50):
        minimum = 0
        maximum = self.video.height-1

        x_start_left= border
        x_start_right=self.video.width - border
        x_offset_step = 50
        y = self.search_height - 20 #Start searching below the search height
        
        left_list = []
        right_list = []
        for i in range(redudancy):
            x_left = x_start_left + i*x_offset_step
            x_right = x_start_right - i*x_offset_step

            local_left = self.video.find_y_edge(segmented_image,x_left,y,direction='Down',minimum=minimum,maximum=maximum)
            local_right = self.video.find_y_edge(segmented_image,x_right,y,direction='Down',minimum=minimum,maximum=maximum)
            
            left_list.append(local_left)
            right_list.append(local_right)

        left = min(left_list)
        right = min(right_list)
        
        if right>left:
            self.init_direction = 'Left'
        elif right<left:
            self.init_direction = 'Right'
        else:
            self.init_direction = 'Left'
            warnings.warn('left and right edge are at the same height, this is unexpected. Searching to the left anyway')
        return min([left,right]), max([left,right])

    def find_jet_lag_frame(self,frame_id,save_frame=False):
        frame = self.video.get_frame(frame_id)
        frame = self.video.rotate_frame(frame) #correction, same as calibration seq

        if self.video.debug.jetlag: 
            print('find jet lag frame')
            self.video.show_frame(frame)

        segmented_image = self.segmentation(frame) #Segmentation of the image
        self.find_jet_direction(segmented_image) #determine left or right search
        self.find_JCP(segmented_image) # Jet Contact Point
        self.calculate_jet_lag() #Convert into unit

        if self.video.debug.CLI: print("Jet Lag: ",round(self.JetLag,2), "[mm]")

        if self.video.debug.annotate: annotated_frame = self.annotate_image(segmented_image)
        if save_frame:
            annotated_frame = self.annotate_image(segmented_image,mode='save')

            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

            cv2.imwrite(self.output_folder+'frame'+str(frame_id)+'.png', annotated_frame)
            cv2.imwrite(self.output_folder+'original_frame'+str(frame_id)+'.png', frame)
        return self.JetLag

    def find_jet_lag_video(self,frame_range=None,save_frame=True,return_extended_output=False):
        if frame_range == None: frame_range = range(1,int(self.video.total_frames))
        
        jet_lag_video = []
        for frame_id in frame_range:
            jet_lag = self.find_jet_lag_frame(frame_id,save_frame=save_frame)
            jet_lag_video.append(jet_lag)
        
        if return_extended_output:
            return (
                frame_range,jet_lag_video,
                self.nozzle_diameter_pixels,
                self.nozzle_centre,
                self.bottom_nozzle_y,
                self.CP_height)
        return frame_range,jet_lag_video

    def set_output_folder(self,location):
        self.output_folder = location


        

