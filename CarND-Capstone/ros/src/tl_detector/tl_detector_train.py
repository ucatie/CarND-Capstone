#!/usr/bin/env python
import rospy
import glob
import os
import time
import image_geometry
import cv2
import math
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.feature import hog
import scipy.misc
import pickle
import light_classification.feature_detection as fd

# NOTE: the next import is only valid for scikit-learn version <= 0.17
#from sklearn.cross_validation import train_test_split
# for scikit-learn >= 0.18 use:


class TLDetector_Train(object):
    def __init__(self):
        rospy.init_node('tl_detector_train')

        self.run_dir = rospy.get_param('/run_dir')
        rospy.loginfo("run_dir:%s",self.run_dir)
        
        self.ground_truth_dir = os.path.join(self.run_dir, rospy.get_param('~ground_truth_dir'))
        rospy.loginfo("ground_truth_dir:%s",self.ground_truth_dir)
            
        self.task = rospy.get_param('~task','best')
        rospy.loginfo("task:%s",self.task)
        
        self.SVC_PATH =  os.path.join(self.run_dir,rospy.get_param('~SVC_PATH','svc.p'))       

#        rospy.spin()

    #answer a sample or all database images
    def displayDatabaseSample(self):    
      # Read in cars and notcars
      red = []
      green = []
      yellow = []

      red_gt_images = glob.glob(os.path.join(os.path.join(self.ground_truth_dir,'0'),'*.jpg'))
      for image in red_gt_images:
        red.append(image)
    
      green_gt_images = glob.glob(os.path.join(os.path.join(self.ground_truth_dir,'2'),'*.jpg'))
      for image in  green_gt_images:
        green.append(image)
        
      yellow_gt_images = glob.glob(os.path.join(os.path.join(self.ground_truth_dir,'1'),'*.jpg'))
      for image in  yellow_gt_images:
        yellow.append(image)
        
      np.random.seed(1)
      sample_size = 64
      random_indizes = np.arange(sample_size)
      np.random.shuffle(random_indizes)
    
      cars = np.array(cars)[random_indizes]
      notcars = np.array(notcars)[random_indizes]
    
      #show one image for 20 samples
      row = 0
      col = 0
      fig = plt.figure()
      size = 6
      gs = gridspec.GridSpec(size, size)
    #  fig.subplots_adjust(top=1.5)
      for j in range(4):
       if j == 0:
         data = red
         title = "red"
       elif j == 1: 
         data = green
         title = "green"
       elif j == 2: 
         data = yellow
         title = "yellow"
        
       for i in range(int(size*size/2)):
        image = mpimg.imread(data[i])
        a = plt.subplot(gs[row, col])
        a.imshow(image.copy())
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
        a.set_title(title)
        fig.add_subplot(a)
        col += 1                   
        if col == size:
          col = 0
          row += 1
                
      fig = plt.gcf()
      fig.savefig("samples.png") 
      plt.show()
    
    #answer a sample or all database images
    def readDatabase(self, reducedSamples):    
      # Read in arrays
      red = []
      green = []
      yellow = []
      unknown = []
#      rospy.loginfo("red path:%s",os.path.join(self.ground_truth_dir,'*_0.jpg'))
      red = glob.glob(os.path.join(os.path.join(self.ground_truth_dir,'0'),'*.jpg'))  
      green = glob.glob(os.path.join(os.path.join(self.ground_truth_dir,'2'),'*.jpg'))
      yellow = glob.glob(os.path.join(os.path.join(self.ground_truth_dir,'1'),'*.jpg'))
      
      image = mpimg.imread(red[0])
      rospy.loginfo("min: %s max: %s",np.min(image[0]),np.max(image[0]))
      
      if reducedSamples:    
        print("read sample database")
        np.random.seed(1)
      # Reduce the sample size for fast testing
        sample_size = 64

        random_indizes = np.arange(min(sample_size, len(red)))
        np.random.shuffle(random_indizes)
        red = np.array(red)[random_indizes]
        
        random_indizes = np.arange(min(sample_size, len(green)))
        np.random.shuffle(random_indizes)
        green = np.array(green)[random_indizes]
        
        random_indizes = np.arange(min(sample_size, len(yellow)))
        np.random.shuffle(random_indizes)
        yellow = np.array(yellow)[random_indizes]
        
      else:
        rospy.loginfo("read database")
    
      rospy.loginfo("red: {0} green {1} yellow {2}".format(len(red),len(green),len(yellow)))
      return (red,green,yellow)
    
    #answer a list of param for optimization
    def getParamlist3(self):
      params = (
        ("YCrCb","ALL",True,True,True,3),
        ("YCrCb","0,1",True,True,True,3),
        ("YCrCb","0,1",False,True,True,3),
        ("YCrCb","0,1",True,False,True,3),
        ("YCrCb","0,1",False,False,True,3),
        ("YCrCb","0,1",True,True,False,3),
        ("YCrCb","0,1",False,True,False,3),
        ("YCrCb","0,1",True,False,False,3),
        ("YCrCb","0,1",False,False,False,3))
      return params
    
    def getParamlist2(self):
      params = (
        ("YUV","ALL",False,False,True,3),
        ("YUV",0,False,False,True,3),
        ("YUV",1,False,False,True,3),
        ("YUV",2,False,False,True,3),
        ("YUV","0,1",False,False,True,3),
        ("YUV","0,2",False,False,True,3),
        ("YUV","1,2",False,False,True,3),
        ("YCrCb","All",False,False,True,3),
        ("YCrCb",0,False,False,True,3),
        ("YCrCb",1,False,False,True,3),
        ("YCrCb",2,False,False,True,3),
        ("YCrCb","0,1",False,False,True,3),
        ("YCrCb","0,2",False,False,True,3),
        ("YCrCb","1,2",False,False,True,3))
      return params
    
    def getParamlist1(self):
      params = (
        ("RGB","ALL",True,True,True,3),
        ("RGB","ALL",False,True,True,3),
        ("RGB","ALL",True,False,True,3),
        ("RGB","ALL",False,False,True,3),
        ("RGB","ALL",True,True,False,3),
        ("RGB","ALL",False,True,False,3),
        ("RGB","ALL",True,False,False,3),
        ("HSV","ALL",True,True,True,3),
        ("HSV","ALL",False,True,True,3),
        ("HSV","ALL",True,False,True,3),
        ("HSV","ALL",False,False,True,3),
        ("HSV","ALL",True,True,False,3),
        ("HSV","ALL",False,True,False,3),
        ("HSV","ALL",True,False,False,3),
        ("LUV","ALL",True,True,True,3),
        ("LUV","ALL",False,True,True,3),
        ("LUV","ALL",True,False,True,3),
        ("LUV","ALL",False,False,True,3),
        ("LUV","ALL",True,True,False,3),
        ("LUV","ALL",False,True,False,3),
        ("LUV","ALL",True,False,False,3),
        ("HLS","ALL",True,True,True,3),
        ("HLS","ALL",False,True,True,3),
        ("HLS","ALL",True,False,True,3),
        ("HLS","ALL",False,False,True,3),
        ("HLS","ALL",True,True,False,3),
        ("HLS","ALL",False,True,False,3),
        ("HLS","ALL",True,False,False,3),
        ("YUV","ALL",True,True,True,3),
        ("YUV","ALL",False,True,True,3),
        ("YUV","ALL",True,False,True,3),
        ("YUV","ALL",False,False,True,3),
        ("YUV","ALL",True,True,False,3),
        ("YUV","ALL",False,True,False,3),
        ("YUV","ALL",True,False,False,3),
        ("YCrCb","ALL",True,True,True,3),
        ("YCrCb","ALL",False,True,True,3),
        ("YCrCb","ALL",True,False,True,3),
        ("YCrCb","ALL",False,False,True,3),
        ("YCrCb","ALL",True,True,False,3),
        ("YCrCb","ALL",False,True,False,3),
        ("YCrCb","ALL",True,False,False,3))
      return params
    
    #train a list of parameters
    def trainParamlist(self, red,green,yellow,params):
    
      results = {}
      #train the list
      for i in range(len(params)):
        key = params[i]
        try:
            (scaler,svc,accuracy) = self.train(params[i],red,green,yellow)
            results[key] = accuracy
        except ValueError: 
            rospy.loginfo("error in train {0}".format(key))
            results[key] = 0
    
      #print result as wiki table
      rospy.loginfo("|Color space| Channels | Spatial | Histogram | HOG | Accuracy |")
      best = sorted(results, key=results.get, reverse=True)[0:min(4,len(results))]
      for i in range(len(params)):
        key = params[i]
        accuracy = results[key]
        
        (color_space,hog_channel,spatial_feat,hist_feat,hog_feat,cell_per_block) = key
    
        if key in best:
          rospy.loginfo("| %s | %s | %s | %s | %s | ** %s ** |",color_space,hog_channel,spatial_feat,hist_feat,hog_feat,accuracy)
        else:
          rospy.loginfo("| %s | %s | %s | %s | %s | %s |",color_space,hog_channel,spatial_feat,hist_feat,hog_feat,accuracy)
    
    #create normalized, randomly shuffelded test and train data for a parameter set    
    def getFeatures(self, param,red,green,yellow):
      print(param)
    
      (color_space,hog_channel,spatial_feat,hist_feat,hog_feat,cell_per_block) = param
    
      orient = 9  # HOG orientations
      pix_per_cell = 8 # HOG pixels per cell
      spatial_size = (16, 16) # Spatial binning dimensions
      hist_bins = 16    # Number of histogram bins
    
      red_features = fd.extractFeatures(red, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat,feature_vec=True)
    
      green_features = fd.extractFeatures(green, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat,feature_vec=True)

      yellow_features = fd.extractFeatures(yellow, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat,feature_vec=True)
    
      X = np.vstack((red_features, green_features, yellow_features)).astype(np.float64)      
      
    # Fit a per-column scaler
      X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
      scaled_X = X_scaler.transform(X)
    # Define the labels vector
      y = np.hstack((np.zeros(len(red_features)), np.ones(len(green_features))*2,np.ones(len(yellow_features))))
    #Split up data into randomized training and test sets
      rand_state = 1
      X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    
      rospy.loginfo(' using orientations: %s pixels per cell: %s cells per block %s -> feature vector length: %s',orient,pix_per_cell,cell_per_block,len(X_train[0]))
    
      return (X_train,X_test,y_train,y_test,X_scaler)
        
    #train once
    def train(self, param,red,green,yellow):
        
      (X_train, X_test, y_train, y_test,X_scaler) = self.getFeatures(param,red,green,yellow)
    
    #train
      svc = SGDClassifier(fit_intercept=False, loss="squared_hinge", n_jobs=-1, learning_rate="optimal", penalty="elasticnet", class_weight="balanced",n_iter=10, alpha=0.01)
      svc.fit(X_train, y_train)
    
    #get the accuracy      
      accuracy = round(svc.score(X_test, y_test), 4)
      rospy.loginfo('Test Accuracy of SVC = %s', accuracy)
    
      return (X_scaler,svc,accuracy)
    
    #load a pickle file and return the model dictionary  containg the keys X_scaler and svc
    def getModelData(self):
      data = {}
      #load trained svc
      with open(SVC_PATH, "rb") as f:
        data = pickle.load(f)
      return data
    
    #search optimal classifier paramter
    def gridSearch(self, red,green,yellow,unknown,param):
      (X_train, X_test, y_train, y_test, X_scaler) = getFeatures(param,red,green,yellow)
    
      scores = ['precision', 'recall']
    
      # Set the parameters space
      tuned_parameters = [{'loss':["hinge","modified_huber","squared_hinge"],'alpha': [0.00001,0.0001,0.001,0.01],"penalty":["l1","l2","elasticnet"]}]
    
      scores = ['precision', 'recall']
    
      for score in scores:
        rospy.loginfo("# Tuning hyper-parameters for %s" % score)
        rospy.loginfo()
    
        clf = GridSearchCV(SGDClassifier(shuffle=True, fit_intercept=False, n_jobs=-1, learning_rate="optimal", penalty="l2", class_weight="balanced",n_iter=5), tuned_parameters, cv=5, scoring='%s_macro' % score)
        clf.fit(X_train, y_train)
    
        rospy.loginfo("Best parameters set found on development set:")
        rospy.loginfo()
        rospy.loginfo(clf.best_params_)
        rospy.loginfo()
        rospy.loginfo("Grid scores on development set:")
        rospy.loginfo()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
          rospy.loginfo("%0.3f (+/-%0.03f) for %r".format(mean, std * 2, params))
          rospy.loginfo()
    
          rospy.loginfo("Detailed classification report:")
          rospy.loginfo()
          rospy.loginfo("The model is trained on the full development set.")
          rospy.loginfo("The scores are computed on the full evaluation set.")
          rospy.loginfo()
          y_true, y_pred = y_test, clf.predict(X_test)
          rospy.loginfo(classification_report(y_true, y_pred))
          rospy.loginfo()

    def run_task(self):      
    
      if self.task == 'displaySample':
        self.displayDatabaseSample()
    
      elif self.task == 'trainList1':
        #read a sample
        (red,green,yellow) = self.readDatabase(True)
        #train a list of params for optimization
        self.trainParamlist(red,green,yellow,self.getParamlist1())
    
      elif self.task == 'trainList2':
        #read a sample
        (red,green,yellow) = self.readDatabase(True)
        #train a list of params for optimization
        self.trainParamlist(red,green,yellow,self.getParamlist2())
    
      elif self.task == 'trainList3':
        #read a sample
        (red,green,yellow) = self.readDatabase(True)
        #train a list of params for optimization
        self.trainParamlist(red,green,yellow,self.getParamlist3())
    
      elif self.task == 'best':
        #read all
        (red,green,yellow) = self.readDatabase(False)
        #train the best choice
        param = ("RGB","ALL",True,True,False,3)
        (X_scaler,svc,acccuracy) = self.train(param,red,green,yellow)
    
        #save the calibration in a pickle file
        data = {}
        data["X_scaler"] = X_scaler
        data["svc"] = svc
        data["param"] = param
        with open(self.SVC_PATH, 'wb') as f:
          pickle.dump(data, file=f)
          
        rospy.loginfo("saved trained svc in %s",self.SVC_PATH)    
        
      elif self.task == 'gridSearch':
        #read a sample
        (red,green,yellow,unknown) = self.readDatabase(True)
        #find optimal Classifier parameter
        param = ("YCrCb","0,1",True,False,True,3)
        self.gridSearch(red,green,yellow,param)
    
        
if __name__ == '__main__':
    try:
        tlt = TLDetector_Train()
        tlt.run_task()
        rospy.signal_shutdown("finished")      
        
    except rospy.ROSInterruptException:
        rospy.logerr('Could not run tl detector train node.')
        
        