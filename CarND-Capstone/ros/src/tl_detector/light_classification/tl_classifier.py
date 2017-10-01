import rospy
from styx_msgs.msg import TrafficLight
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import scipy.misc
import pickle
import feature_detection as fd
import tensorflow as tf
import tl_fcn_classifier as fcn
import math


class TLClassifier(object):
    def __init__(self, SVC_PATH, is_simulator, fcn_path):

        #load the model     
        data = self.getModelData(SVC_PATH)
        self.X_scaler = data["X_scaler"]
        self.svc = data["svc"]
        self.param = data["param"]
                
        self.is_simulator = is_simulator        
        rospy.loginfo("param %s",self.param)
        
        self.session = None
        if not is_simulator:
            rospy.loginfo("fcn_path %s",fcn_path)
            self.session =  tf.Session() 
            self.fcn_variables = fcn.load_fcn(self.session,fcn_path)            
  
    #load a pickle file and return the model dictionary  containg the keys X_scaler and svc
    def getModelData(self,SVC_PATH):
      data = {}
      #load trained svc
      with open(SVC_PATH, "rb") as f:
        data = pickle.load(f)
        
      rospy.loginfo("loaded model %s",SVC_PATH)
      return data
  
        #define search parameter for window search.
    def getSearchParam(self,shape):
        search_param = []
        search_param.append((shape[0],0,shape[0],0,shape[1],1))  #simulator
#        search_param.append((64,0,64,0,64,1))  
        return search_param
  
    #the process chain for an image. 
    def processColoredImage(self,image,debug,X_scaler,svc,param):
#        heatmap = gd.data["heatmap"]
 #       gd.data["image_counter"]= gd.data["image_counter"] + 1
        if image is None:
            return None
      
        normImage = image.copy().astype("float32") / 255.0

        windows = []
        result = image
        debugImg = image
    
        search_param = self.getSearchParam(image.shape)
#        all_hot_windows = []
        #  create the windows list sliding ofver the search area
        for i in range(len(search_param)):  
            (size,y_start, y_stop, x_start, x_stop,ov) = search_param[i]
    
            new_win = fd.slideWindow(normImage, x_start_stop=(x_start, x_stop), y_start_stop=(y_start, y_stop), 
                                     xy_window=(size,size), xy_overlap=(ov, ov))
        
            windows += new_win
    
        #  print(windows2)
        (color_space,hog_channel,spatial_feat,hist_feat,hog_feat,cell_per_block) = param
        orient = 9  # HOG orientations
        pix_per_cell = 8 # HOG pixels per cell
        spatial_size = (16, 16) # Spatial binning dimensions
        hist_bins = 16    # Number of histogram bins
        # classify the image parts in the search windows
        hot_windows = fd.searchWindows(normImage, windows, svc, X_scaler, color_space=color_space, 
                          spatial_size=spatial_size, hist_bins=hist_bins, 
                          orient=orient, pix_per_cell=pix_per_cell, 
                          cell_per_block=cell_per_block, 
                          hog_channel=hog_channel, spatial_feat=spatial_feat, 
                          hist_feat=hist_feat, hog_feat=hog_feat)
        
        best_state = -1
        max_count = -1
        for i in range(4):
            if len(hot_windows[i]) > max_count:
                best_state = i
                max_count = len(hot_windows[i])
#                rospy.loginfo("state %s count %s",best_state, max_count)
            elif max_count > 0 and len(hot_windows[i]) == max_count:
                best_state = 4
                max_count = len(hot_windows[i])
#                rospy.loginfo("eqality set to unknown state %s count %s",best_state, max_count)
        
#        rospy.loginfo("best state %s count %s",best_state, max_count)
        return best_state
                                   
    def getSearchParam2(self, shape):
        search_param = []
#        search_param.append((128,256,768,0,1024,0.75))
        if self.is_simulator:
            search_param.append((20,int(shape[0]*0.2),int(shape[0]*0.9),10,shape[1]-10,0.75))  #simulator
        else:
            search_param.append((32,int(shape[0]*0.2),int(shape[0]*0.9),0,shape[1],0.9414))  #real
        return search_param
                              
    def distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])*(pos1[0] - pos2[0]) + (pos1[1] - pos2[1])*(pos1[1] - pos2[1]))
                                       
    counter = 0

    def find_center(self, windows, bitmap):
        #filter windows that have a center in the bitmap
        filtered_windows = []
        shape = (windows[0][1][1]-windows[0][0][1],windows[0][1][0]-windows[0][0][0])
        for window in windows:            
            center = (int(window[0][0] + shape[0]*0.5),int(window[0][1] + shape[1]*0.5))
            if bitmap[center[1],center[0]] > 0:
#                rospy.loginfo("filtered_windows %s",filtered_windows)
                filtered_windows.append(window)        
           
        #find the window that is most center in the window
        max_sum = 0
        best_pos = None
        for window in filtered_windows:
            bsum = np.sum(bitmap[window[0][1]:window[1][1],window[0][0]:window[1][0]])
#            rospy.loginfo("window %s bsum %s",window, bsum)
            if max_sum < bsum:
#                rospy.loginfo("add window %s bsum %s",window, bsum)
                max_sum = bsum
                best_pos = []
                center = (int(window[0][0] + shape[0]*0.5),int(window[0][1] + shape[1]*0.5))
                best_pos.append(center)             
                             
            elif max_sum == bsum:
                center = (int(window[0][0] + shape[0]*0.5),int(window[0][1] + shape[1]*0.5))
                #could be two traffic lights in one image
                if abs(center[0]- best_pos[0][0]) <10 and abs(center[1]- best_pos[0][1]) < 10: 
#                    rospy.loginfo("add window %s bsum %s",window, bsum)
                    best_pos.append(center)             
                             
        if best_pos is None:
            return None
        
        center = np.mean(best_pos,axis = 0).astype(int)
#        rospy.loginfo("center %s ",center)
        
#        if debug:
#            debugImg = fd.drawBoxes(image, filtered_windows, color=(0, 0, 255), thick=1)
#            global counter
#            counter = 1
#            path_to_image = os.path.join("/home/frank/selfdriving/sdc_course/CarND-Capstone/debug","{0}_{1}.jpg".format("debug",counter))
#            scipy.misc.imsave(path_to_image, debugImg)    
        
        return center
        
    #the process chain for an image for the real images
    def find_real_class_position(self,orig_image,debug):

        windows = []
    
        image_shape = (256,256)
        image = scipy.misc.imresize(orig_image, image_shape)
    
        search_param = self.getSearchParam2(image.shape)
#        rospy.loginfo("search %s",search_param)
#        all_hot_windows = []
        #  create the windows list sliding ofver the search area
        for i in range(len(search_param)):  
            (size,y_start, y_stop, x_start, x_stop,ov) = search_param[i]
    
            new_win = fd.slideWindow(image, x_start_stop=(x_start, x_stop), y_start_stop=(y_start, y_stop), 
                                     xy_window=(size,size), xy_overlap=(ov, ov))
        
            windows += new_win
    
        keep_prob, input, logits = self.fcn_variables    
        all_bm = fcn.classify(self.session,keep_prob, input, logits,image)
 #       all_bm = image#dummy
          
#        rospy.loginfo("tl light %s %s",all_bm.shape, np.sum(all_bm))
        bitmap = None
        if np.sum(all_bm) > 50:
#            rospy.loginfo("found tl light")
            bitmap = all_bm;
               
        if bitmap is None:
            return None
        
        center = self.find_center(windows,bitmap)
        if center is None:
            return None
#        rospy.loginfo("found %s",center)

#        rospy.loginfo("best state %s count %s",center, max_count)
        x,y = (int(center[0]*orig_image.shape[1]/256),int(center[1]*orig_image.shape[0]/256))
#        rospy.loginfo("found tl at %s %s",center,(x,y))
        return (x,y)                                   
        
    #the process chain for an image for the simulator 
    def find_sim_class_position(self,image,debug):
#        heatmap = gd.data["heatmap"]
 #       gd.data["image_counter"]= gd.data["image_counter"] + 1
      
        windows = []
        debugImg = image
    
        search_param = self.getSearchParam2(image.shape)
#        all_hot_windows = []
        #  create the windows list sliding ofver the search area
        for i in range(len(search_param)):  
            (size,y_start, y_stop, x_start, x_stop,ov) = search_param[i]
    
            new_win = fd.slideWindow(image, x_start_stop=(x_start, x_stop), y_start_stop=(y_start, y_stop), 
                                     xy_window=(size,size), xy_overlap=(ov, ov))
        
            windows += new_win
    
        bitmap = None   
        green_bm = fd.greenBinaryFromRGB(image,self.is_simulator)  
        if np.sum(green_bm) > 10000:
#            rospy.loginfo("looking for green")
            bitmap = green_bm;
            
        yellow_bm = fd.yellowBinaryFromRGB(image,self.is_simulator)  
        if np.sum(yellow_bm) > 10000:
#            rospy.loginfo("looking for yellow")
            bitmap = yellow_bm;
            
        red_bm = fd.redBinaryFromRGB(image,self.is_simulator)  
        if np.sum(red_bm) > 10000:
#            rospy.loginfo("looking for red")
            bitmap = red_bm;
               
        if bitmap is None:
            return None
        
        #find the window that is most center in the window
        center = self.find_center(windows,bitmap)
        return center                                   
    
    def get_classification(self, image):        
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #implement light color prediction
        
        state = self.processColoredImage(image,False,self.X_scaler, self.svc, self.param)
#        rospy.loginfo("classifier state %s",state)
        return  state       

    def find_classification(self, image):        
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #implement light color prediction
        
        if self.is_simulator:
            pos = self.find_sim_class_position(image,False)
        else:
            pos = self.find_real_class_position(image,False)
            
        if pos is not None:
            return  pos
        return (None,None)       

