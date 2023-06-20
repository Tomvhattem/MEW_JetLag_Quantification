

import numpy as np
import cv2 as cv
import time
import datetime
import os
import serial
import struct
import copy
import collections
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#import functions
#from functions import *
from threading import Thread, Lock
from multiprocessing import Process, Queue
import gc

class VideoReader :
    """
    Class creating a thread responsible for reading frames
    """
    def __init__(self, src = 0, fps=50,maxbuffer=3) :
        self.stream = cv.VideoCapture(src)
        if fps:
            self.stream.set(cv.CAP_PROP_FPS,fps)
        (self.grabbed, self.frame) = self.stream.read()
        self.fps=fps
        self.started = False
        self.stack=[]
        self.maxbuffer=maxbuffer
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print ("Video Reader already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            time.sleep(1./self.fps)
            if grabbed:
                if len(self.stack)>=self.maxbuffer:
                    del self.stack[:]
                    gc.collect()
                self.stack.append(frame)

                self.read_lock.acquire()
                self.grabbed, self.frame = grabbed, frame
                self.read_lock.release()
            else:
                self.started = False

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame
        self.read_lock.release()
        if len(self.stack)!=0:
            #print(len(self.stack))
            self.stack.pop(0)
            #print(len(self.stack))
        return frame

    def stop(self) :
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()
    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

class VideoShower:
    """
    Class creating a thread responsible for showing video
    """
    def __init__(self,frame=None,fps=50,name='Feed'):

        self.frame=frame
        self.fps=fps
        self.name=name
        self.started=False

    def start(self):
        if self.started:
            print("Video Shower already started!!")
            return None
        self.started=True
        self.thread=Thread(target=self.show,args=())
        self.thread.start()
        return self
    def show(self):
        while self.started:
            cv.imshow(self.name, self.frame)
            if cv.waitKey(int(1000*(1/self.fps))) == ord('q'):
                self.started = False

    def stop(self):
        self.started=False
        if self.thread.is_alive():
            self.thread.join()
        cv.destroyAllWindows()
    def __exit__(self, exc_type, exc_value, traceback):
        cv.destroyAllWindows()



class VideoWriter:
    """
    Class responsible for storing video
    """
    def __init__(self,frame=None):
        self.frame=frame
        self.started=False
    def start(self,name,wps):
        if self.started:
            print("Video Writer already started!!")
            return None
        self.started=True
        self.wps = wps
        fourcc=cv.VideoWriter_fourcc(*'XVID')
        self.final_output=cv.VideoWriter(name,fourcc,wps,(self.frame.shape[1],self.frame.shape[0]))
        self.thread=Thread(target=self.write,args=())
        self.thread.start()
        return self
    def write(self):
        while self.started:
            self.final_output.write(self.frame)
            if cv.waitKey(int(1000*(1/self.wps)))==ord('q'):
                self.started=False

    def stop(self):
        self.started=False
        if self.thread.is_alive():
            self.thread.join()