#!/usr/bin/env python
import rospy
import os
import glob
import matplotlib.image as mpimg
from light_classification.tl_classifier import TLClassifier

if __name__ == '__main__':
    rospy.init_node('test_classifier')
    
    run_dir = rospy.get_param('/run_dir')
    rospy.loginfo("run_dir:%s",run_dir)
    
    train_data_dir = os.path.join(run_dir,rospy.get_param('~train_data_dir'))
    rospy.loginfo("train_data_dir:%s",train_data_dir)

    SVC_PATH =  os.path.join(run_dir,rospy.get_param('~SVC_PATH','svc.p'))       
    rospy.loginfo("SVC_PATH:%s",SVC_PATH)
    
    tc = TLClassifier(SVC_PATH)

    images = []
    train_images = glob.glob(os.path.join(train_data_dir,'*.jpg'))
    for image_name in train_images:
        image = mpimg.imread(image_name)
        if image is None:
            continue
        
        state = tc.get_classification(image)
        rospy.loginfo("image_name:%s state %s",image_name, state)
  
