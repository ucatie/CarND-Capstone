import os
import rospy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog 

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True,
                                  block_norm="L2-Hys",
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       block_norm="L2-Hys",
                       visualise=vis, feature_vector=feature_vec)
#        print(features.shape)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def greenBinaryFromRGB(image, is_simulator):
    binary = np.zeros_like(image[:,:,0])
    if is_simulator:
        binary[(image[:,:,0] <= 90) \
            & (image[:,:,1] >= 200) \
            & (image[:,:,2] <=  90)]= 255
    else:
        binary[(image[:,:,0] >= 150) \
            & (image[:,:,1] >= 200) \
            & (image[:,:,2] >= 200) \
            & (image[:,:,0] < 190)]= 255

    return binary  

def redBinaryFromRGB(image, is_simulator):
    binary = np.zeros_like(image[:,:,0])
    if is_simulator:
        binary[(image[:,:,0] >= 200) \
            & (image[:,:,1] <= 120) \
            & (image[:,:,2] <= 120)]= 255
    else:
        binary[(image[:,:,0] >= 250) \
            & (image[:,:,1] >= 250) \
            & (image[:,:,2] >= 100) \
            & (image[:,:,2] < 190)]= 255
    return binary  

def yellowBinaryFromRGB(image, is_simulator):
    binary = np.zeros_like(image[:,:,0])
    if is_simulator:
        binary[(image[:,:,0] >= 200) \
            & (image[:,:,1] >= 200) \
            & (image[:,:,2] <= 90)]= 255
    else:
        binary[(image[:,:,0] >= 250) \
            & (image[:,:,1] >= 250) \
            & (image[:,:,2] >= 100) \
            & (image[:,:,2] < 190)]= 255
    return binary  
            
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extractFeatures(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True,feature_vec=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        normImage = image.copy().astype("float32") / 255.0
        
        
        img_features = singleImgFeatures(normImage,color_space,spatial_size,hist_bins,orient, 
                        pix_per_cell,cell_per_block,hog_channel,
                        spatial_feat,hist_feat,hog_feat,feature_vec)        
        
        features.append(np.concatenate(img_features))
    # Return list of feature vectors
    return features
    

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def singleImgFeatures(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True,feature_vec=True):    
    #1) Define an empty list to receive features
    img_features = []   
#    img= cv2.GaussianBlur(img, (9, 9), 0)    
    
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if isinstance(hog_channel,str):
          if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=feature_vec))      
            if feature_vec:
              hog_features = np.ravel(hog_features)        
          else:
            channels = hog_channel.split(",")
            hog_features = []
            for channel in channels:
                hog_features.append(get_hog_features(feature_image[:,:,int(channel)], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=feature_vec))      
            
            if feature_vec:
              hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=feature_vec)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
#    return np.concatenate(img_features)
    return img_features
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slideWindow(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]    
    yspan = y_start_stop[1] - y_start_stop[0]
    nx_windows = 1
    ny_windows = 1
    # Compute the number of pixels per step in x/y
    # Compute the number of windows in x/y
    nx_pix_per_step = xy_window[0]*(1 - xy_overlap[0])
    if nx_pix_per_step == 0:
        nx_pix_per_step = xy_window[0]
    nx_windows = int((xspan - xy_window[0])/nx_pix_per_step+1)
        
    ny_pix_per_step = xy_window[1]*(1 - xy_overlap[1])
    if ny_pix_per_step == 0:
        ny_pix_per_step = xy_window[1]        
    ny_windows = int((yspan - xy_window[1])/ny_pix_per_step+1)

    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        starty = int(ys*ny_pix_per_step + y_start_stop[0])
        endy = starty + xy_window[1]
 #       rospy.loginfo("y start %s stop %s",(starty),(endy))
        if endy > y_start_stop[1]+1:
          print("reduce endy:",endy, y_start_stop[1],(endy - y_start_stop[1]),xy_window[1])
          continue

        if ys == ny_windows-1 and endy < y_start_stop[1]-1:
          print("extend endy:",endy, y_start_stop[1],(endy - y_start_stop[1]),xy_window[1])
            
        for xs in range(nx_windows):
            # Calculate window position
            startx = int(xs*nx_pix_per_step + x_start_stop[0])
            endx = startx + xy_window[0]
            if endx > x_start_stop[1]+1:
              print("reduce endx:",endx, x_start_stop[1],(endx - x_start_stop[1]),xy_window[0])
              continue
            if xs == nx_windows-1 and endx < x_start_stop[1]-1:
              print("extend endx:",endx, x_start_stop[1],(endx - x_start_stop[1]),xy_window[0])
            # Append window position to list
#            rospy.loginfo("search window %s %s",(startx, starty), (endx, endy))
            
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

image_counter = 1
# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def searchWindows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    global image_counter
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #create 5 state list
    for i in range(5):
        on_windows.append([])
    
    shape = (windows[0][1][1]-windows[0][0][1],windows[0][1][0]-windows[0][0][0])
    #2) Iterate over all windows in the list
    for window in windows:
#            print("center:",center[0],center[1])
#            saveImageAndReturn(bitmap,"/home/frank/selfdriving/sdc_course/CarND-Capstone/ros/bitmap",1)
                 
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], shape)     
        #saveImageAndReturn(test_img,"light",image_counter)
        image_counter = image_counter+1
        #4) Extract features for that window using single_img_features()
        features = singleImgFeatures(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        features = np.concatenate(features)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = int(clf.predict(test_features))
#        rospy.loginfo("window %s prediction %s", window, prediction)
        #7) for each state the window is stored in a list 
        on_windows[prediction].append(window)
    #8) Return windows off detections
    return on_windows
        

# Define a function to draw bounding boxes
def drawBoxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


#for logging bad situations
def saveImageAndReturn(image, name, image_counter):
  imageName = "{0}{1}.jpg".format(name,image_counter)

  path_to_image = os.path.join("/home/frank/selfdriving/sdc_course/CarND-Capstone/ros/data_gt/no_light",imageName)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  cv2.imwrite(path_to_image, image)
  print("stored image:",imageName)
  return image       