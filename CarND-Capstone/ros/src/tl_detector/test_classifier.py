#!/usr/bin/env python
import rospy
import os
import glob
import matplotlib.image as mpimg
from light_classification.tl_classifier import TLClassifier
import cv2
import light_classification.feature_detection as fd
import numpy as np
import scipy.misc

def test_image():
    rospy.init_node('test_classifier')
    
    path1 = "/mnt/hgfs/System-Integration/CarND-Capstone/bag_dump_just_traffic_light/classified/no_light/light1.jpg"
    image = mpimg.imread(path1)
    hist = fd.color_hist(image, nbins=5)
    rospy.loginfo("r min: %s max: %s",np.min(image[0]),np.max(image[0]))
    rospy.loginfo("g min: %s max: %s",np.min(image[1]),np.max(image[1]))
    rospy.loginfo("b min: %s max: %s",np.min(image[2]),np.max(image[2]))
    print(hist)

    path2 = "/mnt/hgfs/System-Integration/CarND-Capstone/bag_dump_just_traffic_light/single/left0573.jpg"
    size=(1024,1024)
    image = mpimg.imread(path2)
    image = cv2.resize(image, size)
    path_to_image = os.path.join("/mnt/hgfs/System-Integration/CarND-Capstone/bag_dump_just_traffic_light/single","test1.jpg")
    scipy.misc.imsave(path_to_image, image)    
    image = image[256:384,0:128]
#    image = cv2.resize(image[256:384][0:128], (128,128))
    path_to_image = os.path.join("/mnt/hgfs/System-Integration/CarND-Capstone/bag_dump_just_traffic_light/single","test2.jpg")
    scipy.misc.imsave(path_to_image, image)    
    
    hist = fd.color_hist(image, nbins=5)
    rospy.loginfo("r min: %s max: %s",np.min(image[0]),np.max(image[0]))
    rospy.loginfo("g min: %s max: %s",np.min(image[1]),np.max(image[1]))
    rospy.loginfo("b min: %s max: %s",np.min(image[2]),np.max(image[2]))

    print(hist)


def run():
    rospy.init_node('test_classifier')
    
    run_dir = rospy.get_param('/run_dir')
    rospy.loginfo("run_dir:%s",run_dir)
    
    train_data_dir = os.path.join(run_dir,rospy.get_param('~train_data_dir'))
    rospy.loginfo("train_data_dir:%s",train_data_dir)

    SVC_PATH =  os.path.join(run_dir,rospy.get_param('~SVC_PATH','svc.p'))       
    rospy.loginfo("SVC_PATH:%s",SVC_PATH)
    
    tc = TLClassifier(SVC_PATH)

    images = []
    size=(1024,1024)
    train_images = glob.glob(os.path.join(train_data_dir,'*.jpg'))
    rospy.loginfo("process %s images in %s",len(train_images), train_data_dir)
    for image_name in train_images:
        image = mpimg.imread(image_name)
        image = cv2.resize(image, size)

        state = tc.find_classification(image)
        rospy.loginfo("image_name:%s state %s",image_name, state)
  
if __name__ == '__main__':
    run()