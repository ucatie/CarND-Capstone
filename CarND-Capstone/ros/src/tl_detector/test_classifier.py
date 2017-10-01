#!/usr/bin/env python
import rospy
import os
import glob
import matplotlib.image as mpimg
import scipy.misc
import cv2

from light_classification.tl_classifier import TLClassifier

if __name__ == '__main__':
    rospy.init_node('test_classifier')
    
    run_dir = rospy.get_param('/run_dir')
    rospy.loginfo("run_dir:%s",run_dir)
    
    test_data_dir = os.path.join(run_dir,rospy.get_param('~test_data_dir'))
    rospy.loginfo("test_data_dir:%s",test_data_dir)

    output_data_dir = os.path.join(run_dir,rospy.get_param('~output_data_dir'))
    rospy.loginfo("output_data_dir:%s",test_data_dir)

    is_simulator = rospy.get_param('~is_simulator')
    rospy.loginfo("is_simulator:%s",is_simulator)
        
    SVC_PATH =  os.path.join(run_dir,rospy.get_param('~SVC_PATH','svc.p'))       
    rospy.loginfo("SVC_PATH:%s",SVC_PATH)
        
    FCN_PATH =  os.path.join(run_dir,rospy.get_param('~FCN_PATH','fcn'))       
    rospy.loginfo("FCN_PATH:%s",FCN_PATH)
    
    create_small_images = True

    tc = TLClassifier(SVC_PATH,is_simulator,FCN_PATH)

    images = []
    test_images = glob.glob(os.path.join(test_data_dir,'*.jpg'))
    
    counter = 1
    for image_name in test_images:
        image = mpimg.imread(image_name)
        image = cv2.resize(image, (600, 800), interpolation = cv2.INTER_AREA)
        
        state = tc.find_classification(image)
        if state[0] is None:
            rospy.logwarn("image_name:%s not classified ",image_name)
            continue
            
        x,y = (int(state[0]),int(state[1]))
        rospy.loginfo("image_name:%s state %s -> %s",image_name, state, (x,y))
        
        if create_small_images:
            small_image = scipy.misc.imresize(image[y-32:y+32,x-32:x+32], (64,64))
            if not is_simulator:
                if x < 32:
                    x = 32    
                small_image = scipy.misc.imresize(image[y-64:y+64,x-32:x+32], (128,128))
            path_to_image = os.path.join(run_dir,"{0}/{1}.jpg".format(output_data_dir,counter))
            scipy.misc.imsave(path_to_image, small_image)    
            counter=counter+1

            
  
