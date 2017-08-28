import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from skimage.feature import hog
import scipy.misc
import pickle
import feature_detection as fd

# NOTE: the next import is only valid for scikit-learn version <= 0.17
#from sklearn.cross_validation import train_test_split
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
    
#answer a sample or all database images
def displayDatabaseSample():    
  # Read in cars and notcars
  red = []
  green = []
  yellow = []
  unknown = []
  red_gt_images = glob.glob('data_gt/*_0.jpg')
  for image in red_gt_images:
    red.append(image)

  green_gt_images = glob.glob('data_gt/*_2.jpg')
  for image in  green_gt_images:
    green.append(image)
	
  yellow_gt_images = glob.glob('data_gt/*_1.jpg')
  for image in  yellow_gt_images:
    yellow.append(image)
	
  unknown = glob.glob('data_gt/*_4.jpg')
  for image in  unknown:
    unknown.append(image)
	
  np.random.seed(1)
  sample_size = 32
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
   elif j == 3:
     data = unknown
     title = "unknown"
    
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
def readDatabase(reducedSamples):    
  # Read in arrays
  red = []
  green = []
  yellow = []
  unknown = []
  red_gt_images = glob.glob('data_gt/*_0.jpg')
  for image in red_gt_images:
    red.append(image)

  green_gt_images = glob.glob('data_gt/*_2.jpg')
  for image in  green_gt_images:
    green.append(image)
	
  yellow_gt_images = glob.glob('data_gt/*_1.jpg')
  for image in  yellow_gt_images:
    yellow.append(image)
	
  unknown = glob.glob('data_gt/*_4.jpg')
  for image in  unknown:
    unknown.append(image)

  image = mpimg.imread(red[0])
  rospy.loginfo("image:",np.min(image[0])," max:",np.max(image[0]))
  
  if reducedSamples:    
    rospy.loginfo("read sample database")
    np.random.seed(1)
  # Reduce the sample size for fast testing
    sample_size = 100
    random_indizes = np.arange(sample_size)
    np.random.shuffle(random_indizes)

    red = np.array(red)[random_indizes]
    green = np.array(green)[random_indizes]
    yellow = np.array(yellow)[random_indizes]
    unknown = np.array(unknown)[random_indizes]
	
  else:
    rospy.loginfo("read database")

  rospy.loginfo("red: {0} green {1} yellow {2} unknown {3}".format(len(red),len(green),len(yellow),len(unknown)))
  return (red,green,yellow,unknown)

#answer a list of param for optimization
def getParamlist3():
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

def getParamlist2():
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

def getParamlist1():
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
def trainParamlist(red,green,yellow,unknown,params):

  results = {}
  #train the list
  for i in range(len(params)):
    key = params[i]
    try:
      (scaler,svc,accuracy) = train(params[i],red,green,yellow,unknown)
      print(key, accuracy)
      results[key] = accuracy
    except ValueError: 
      print("error in train {0}".format(key))
      results[key] = 0

  #print result as wiki table
  print("|Color space| Channels | Spatial | Histogram | HOG | Accuracy |")
  best = sorted(results, key=results.get, reverse=True)[0:min(4,len(results))]
  for i in range(len(params)):
    key = params[i]
    accuracy = results[key]
    print(key, accuracy)
    
    (color_space,hog_channel,spatial_feat,hist_feat,hog_feat) = key

    if key in best:
      print("|",color_space,"|",hog_channel,"|",spatial_feat,"|",hist_feat,"|",hog_feat,"| **",accuracy,"** |")
    else:
      print("|",color_space,"|",hog_channel,"|",spatial_feat,"|",hist_feat,"|",hog_feat,"|",accuracy,"|")

#create normalized, randomly shuffelded test and train data for a parameter set    
def getFeatures(param,red,green,yellow,unknown):
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
  print(np.shape(red_features))

  green_features = fd.extractFeatures(green, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat,feature_vec=True)
  print(np.shape(green_features))
  yellow_features = fd.extractFeatures(green, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat,feature_vec=True)
  print(np.shape(yellow_features))
  unknown_features = fd.extractFeatures(red, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat,feature_vec=True)

  print(np.shape(unknown_features))
  X = np.vstack((red, green)).astype(np.float64)      
  print(X.shape)
  X = np.vstack((red, green, yellow, unknown)).astype(np.float64)      
  print(X.shape)
# Fit a per-column scaler
  X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
  scaled_X = X_scaler.transform(X)
# Define the labels vector
  y = np.hstack((np.zeros(len(red_features)), np.ones(len(green_features))*2,np.ones(len(yellow_features)),np.ones(len(unknown_features))*4))
#Split up data into randomized training and test sets
  rand_state = 1
  X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

  print(' using:',orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block -> feature vector length:', len(X_train[0]))

  return (X_train,X_test,y_train,y_test,X_scaler)
    
#train once
def train(param,red,green,yellow,unknown):
    
  (X_train, X_test, y_train, y_test,X_scaler) = getFeatures(param,red,green,yellow,unknown)

#train
  svc = SGDClassifier(fit_intercept=False, loss="squared_hinge", n_jobs=-1, learning_rate="optimal", penalty="elasticnet", class_weight="balanced",n_iter=10, alpha=0.01)
  svc.fit(X_train, y_train)

#get the accuracy      
  accuracy = round(svc.score(X_test, y_test), 4)
  print('Test Accuracy of SVC = ', accuracy)

  return (X_scaler,svc,accuracy)

SVC_PATH = "./svc.p"
#load a pickle file and return the model dictionary  containg the keys X_scaler and svc
def getModelData():
  data = {}
  #load trained svc
  with open(SVC_PATH, "rb") as f:
    data = pickle.load(f)
  return data

#search optimal classifier paramter
def gridSearch(red,green,yellow,unknown,param):
  (X_train, X_test, y_train, y_test, X_scaler) = getFeatures(param,red,green,yellow,unknown)

  scores = ['precision', 'recall']

  # Set the parameters space
  tuned_parameters = [{'loss':["hinge","modified_huber","squared_hinge"],'alpha': [0.00001,0.0001,0.001,0.01],"penalty":["l1","l2","elasticnet"]}]

  scores = ['precision', 'recall']

  for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SGDClassifier(shuffle=True, fit_intercept=False, n_jobs=-1, learning_rate="optimal", penalty="l2", class_weight="balanced",n_iter=5), tuned_parameters, cv=5, scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
      print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
      print()

      print("Detailed classification report:")
      print()
      print("The model is trained on the full development set.")
      print("The scores are computed on the full evaluation set.")
      print()
      y_true, y_pred = y_test, clf.predict(X_test)
      print(classification_report(y_true, y_pred))
      print()
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train vehicle detection')
  parser.add_argument('task', type=str,
   help='task to execute. (trainList1,trainList2,trainList3,gridSearch,best)')
  args = parser.parse_args()
  task = args.task

  if task == 'displaySample':
    displayDatabaseSample()

  elif task == 'trainList1':
    #read a sample
    (red,green,yellow,unknown) = self.readDatabase(True)
    #train a list of params for optimization
    self.trainParamlist(red,green,yellow,unknown,self.getParamlist1())

  elif task == 'trainList2':
    #read a sample
    (red,green,yellow,unknown) = self.readDatabase(True)
    #train a list of params for optimization
    self.trainParamlist(red,green,yellow,unknown,self.getParamlist2())

  elif task == 'trainList3':
    #read a sample
    (red,green,yellow,unknown) = self.readDatabase(False)
    #train a list of params for optimization
    self.trainParamlist(red,green,yellow,unknown,self.getParamlist3())

  elif task == 'best':
    #read all
    (red,green,yellow,unknown) = self.readDatabase(False)
    #train the best choice
    param = ("YCrCb","0,1",True,False,True,3)
    (X_scaler,svc,acccuracy) = self.train(param,red,green,yellow,unknown)

    #save the calibration in a pickle file
    data = {}
    data["X_scaler"] = X_scaler
    data["svc"] = svc
    data["param"] = param
    with open(SVC_PATH, 'wb') as f:
      pickle.dump(data, file=f)    
    
  elif task == 'gridSearch':
    #read a sample
    (red,green,yellow,unknown) = self.readDatabase(True)
    #find optimal Classifier parameter
    param = ("YCrCb","0,1",True,False,True,3)
    self.gridSearch(red,green,yellow,unknown,param)

      