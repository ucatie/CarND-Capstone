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


class TLClassifier(object):
    def __init__(self, SVC_PATH):

        #load the model     
        data = self.getModelData(SVC_PATH)
        self.X_scaler = data["X_scaler"]
        self.svc = data["svc"]
        self.param = data["param"]        
        rospy.loginfo("param %s",self.param)
  
    #load a pickle file and return the model dictionary  containg the keys X_scaler and svc
    def getModelData(self,SVC_PATH):
      data = {}
      #load trained svc
      with open(SVC_PATH, "rb") as f:
        data = pickle.load(f)
        
      rospy.loginfo("loaded model %s",SVC_PATH)
      return data
  
        #define search parameter for window search.
    def getSearchParam(self):
        search_param = []
        search_param.append((256,0,256,0,256,1))  
        return search_param
  
    #the process chain for an image. 
    def processColoredImage(self,image,debug,X_scaler,svc,param):
#        heatmap = gd.data["heatmap"]
 #       gd.data["image_counter"]= gd.data["image_counter"] + 1
      
        normImage = image.copy().astype("float32") / 255.0

        windows = []
        result = image
        debugImg = image
    
        search_param = self.getSearchParam()
#        all_hot_windows = []
        #  create the windows list sliding ofver the search area
        for i in range(len(search_param)):  
            (size,y_start, y_stop, x_start, x_stop,ov) = search_param[i]
    
            new_win = fd.slideWindow(normImage, x_start_stop=(x_start, x_stop), y_start_stop=(y_start, y_stop), 
                                     xy_window=(size,size), xy_overlap=(ov, ov))
        
            windows += new_win
    
 #       if smooth:
 #           windows += heatmap.getSearchAreas(normImage)
 #       if debug:
 #           debugImg = fd.drawBoxes(debugImg, windows, color=(255,255,255), thick=(1))  
 #       else:
        #static mode for test images  
#            heatmap = hm.Heatmap(1)
       
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
                          hist_feat=hist_feat, hog_feat=hog_feat,hardNegative=False)
        
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
                                   
#        if debug:
            #draw the search window grid for debugging                    
#            debugImg = fd.drawBoxes(debugImg, windows, color=(0,255,255), thick=(2))  
            #draw the windows that got a match for debugging                      
#            debugImg = fd.drawBoxes(debugImg, hot_windows, color=(0, 0, 255), thick=1)                    
    
      #update the heatmap    
#        heatmap.update((image.shape[0],image.shape[1]),hot_windows)
        
#        if smooth:
            #average the heatmap over the history of updates  
#            heatmap.average()
#            heatmap.calcBoxes(result)
#            result = heatmap.drawLabeledBoxes(result,(0,255,0),False)
#            if debug:
#                debugImg = heatmap.drawLabeledBoxes(debugImg,(0,255,0),True)
#            else:
#                heatmap.calcBoxes(result)
#            result = heatmap.drawLabeledBoxes(result,(255,0,0),False)
#            if debug:
#                debugImg = heatmap.drawLabeledBoxes(debugImg,(255,0,0),True)
        
#            if debug:
#                if not imageName is None:
#                    path_to_image = os.path.join("output_images","{0}_{1}.jpg".format(imageName,"debug"))
#                    scipy.misc.imsave(path_to_image, debugImg)    
        
#        if debug:
#            return debugImg
#        else:
#            return result
    
  
    def get_classification(self, image):        
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #implement light color prediction
        
        state = self.processColoredImage(image,False,self.X_scaler, self.svc, self.param)
        rospy.loginfo("classifier state %s",state)
        return  state       

