#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
import tensorflow as tf
import vgg_helper
import warnings
from distutils.version import LooseVersion
import time
import csv
import numpy as np
import rospy

# Check TensorFlow Version
#assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))
tf.initialize_all_variables
# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
   
   
def load_fcn(sess, fcn_path):
    """
    Load Trained FCN Model into TensorFlow.
    :param sess: TensorFlow Session
    :param fcn_path: Path to fcn folder"
    :return: Tuple of Tensors from FCN model (image_input, keep_prob, logits)
    """
    # Download pretrained vgg model
#    vgg_helper.maybe_download_trained_fcn(fcn_path)

    model = tf.train.import_meta_graph(os.path.join(fcn_path, 'model.meta'))
    rospy.loginfo("restored fcn meta graph %s",os.path.join(fcn_path, 'model.meta'))
    graph = tf.get_default_graph()

    model.restore(sess, tf.train.latest_checkpoint(fcn_path))
    rospy.loginfo("restored fcn model %s",fcn_path)
        
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    input = graph.get_tensor_by_name('image_input:0')   
    logits = graph.get_tensor_by_name('logits:0')
    
    return keep_prob, input, logits

#expects the tsaved model in the fcn folder
def test():
    num_classes = 2
    image_shape = (256, 256)
    home_dir = '/home/frank/selfdriving/sdc_course/CarND-Capstone'

    fcn_path = os.path.join(home_dir, 'fcn')
    data_dir = os.path.join(home_dir, 'bag_dump_just_traffic_light')
    runs_dir = os.path.join(home_dir, 'runs')
    
    input = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
#    output = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
 #   feed_dict ={input,output}
    
    batch_size = 1
    with tf.Session() as sess:
        
        model = tf.train.import_meta_graph(os.path.join(fcn_path, 'model.meta'))
        graph = tf.get_default_graph()
#        for v in graph.get_operations():
#            print(v.values())
           
        model.restore(sess, tf.train.latest_checkpoint(fcn_path))
        print("restored fcn model!".format(fcn_path))    
        
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        input = graph.get_tensor_by_name('image_input:0')   
        logits = graph.get_tensor_by_name('logits:0')
        
        #Save inference data using vgg_helper.save_inference_samples
        vgg_helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input)

def classify(session,keep_prob,input,logits,image):
    
        im_softmax = session.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, input: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image.shape[0], image.shape[1])
#        rospy.loginfo("fcn classified %s",im_softmax)
        
        segmentation = (im_softmax > 0.5).reshape(image.shape[0], image.shape[1], 1)        
#        rospy.loginfo("fcn segmentation %s",segmentation)

        return segmentation
    
if __name__ == '__main__':
    test()
