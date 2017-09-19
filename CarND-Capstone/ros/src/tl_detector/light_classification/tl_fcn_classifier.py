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

# Check TensorFlow Version
#assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


   
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # Download pretrained vgg model
#    vgg_helper.maybe_download_pretrained_vgg('./data')

    if not tf.saved_model.loader.maybe_saved_model_directory(vgg_path):
        print("{0} contains NOT a saved model!".format(vgg_path))

    vgg_tag = 'vgg16'    
    model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg_graph = tf.get_default_graph()
    
    print("loaded saved vgg16 model!".format(vgg_path))

    vgg_input_tensor = vgg_graph.get_tensor_by_name('image_input:0')   
    vgg_keep_prob_tensor = vgg_graph.get_tensor_by_name('keep_prob:0')
    vgg_layer3_out_tensor = vgg_graph.get_tensor_by_name('layer3_out:0')
    vgg_layer4_out_tensor = vgg_graph.get_tensor_by_name('layer4_out:0')
    vgg_layer7_out_tensor = vgg_graph.get_tensor_by_name('layer7_out:0')
    
    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.
    Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # FCN-8 - Decoder
    # To build the decoder portion of FCN-8, we'll upsample the input to the
    # original image size.  The shape of the tensor after the final
    # convolutional transpose layer will be 4-dimensional:
    #    (batch_size, original_height, original_width, num_classes).
#    print('vgg_layer7_out Shape: {}'.format(vgg_layer7_out.get_shape()))
#    print('vgg_layer4_out Shape: {}'.format(vgg_layer4_out.get_shape()))
#    print('vgg_layer3_out Shape: {}'.format(vgg_layer3_out.get_shape()))
    
    # making sure the resulting shape has the num_class size in the 4th dimension
    vgg_layer7 = tf.layers.conv2d(
        vgg_layer7_out, num_classes, kernel_size=1, name='vgg_layer7')
    vgg_layer4 = tf.layers.conv2d(
        vgg_layer4_out, num_classes, kernel_size=1, name='vgg_layer4')
    vgg_layer3 = tf.layers.conv2d(
        vgg_layer3_out, num_classes, kernel_size=1, name='vgg_layer3')

#    print('vgg_layer7 Shape: {}'.format(vgg_layer7.get_shape()))
#    print('vgg_layer4 Shape: {}'.format(vgg_layer4.get_shape()))
#    print('vgg_layer3 Shape: {}'.format(vgg_layer3.get_shape()))
        
    fcn_layer1 = tf.layers.conv2d_transpose(vgg_layer7, num_classes, kernel_size=4, 
                            strides=(2, 2), padding='same', name='fcn_layer1')    
#    print('fcn_layer1 Shape: {}'.format(fcn_layer1.get_shape()))    

    fcn_layer2 = tf.add(fcn_layer1, vgg_layer4, name='fcn_layer2')

    fcn_layer3 = tf.layers.conv2d_transpose(fcn_layer2, num_classes, kernel_size=4, 
                            strides=(2, 2),padding='same', name='fcn_layer3')
#    print('fcn_layer3 Shape: {}'.format(fcn_layer3.get_shape()))    

    fcn_layer4 = tf.add(fcn_layer3, vgg_layer3, name='fcn_layer4')
    
    fcn_output = tf.layers.conv2d_transpose(fcn_layer4, num_classes, kernel_size=16, 
                            strides=(8, 8), padding='same', name='fcn_layer4')
#    print('fcn_output Shape: {}'.format(fcn_output.get_shape()))    

    # return the final fcn output
    return fcn_output


def get_trainable_variables():
    print('all trainable_variables:')
    for var in tf.trainable_variables():
        print(var)

    var_list = []
    var_names = ['Variable:0',
            'fc7/weights:0',
            'fc7/biases:0',                 
            'fcn_layer1/kernel:0',
            'fcn_layer1/bias:0' ,
            'fcn_layer3/kernel:0',
            'fcn_layer3/bias:0' ,
            'fcn_layer4/kernel:0',
            'fcn_layer4/bias:0']
            
    for var in tf.trainable_variables():
        if var.name in var_names:
            var_list.append(var)
        
    print('')
    print('used trainable_variables:')
    for var in var_list:
        print(var)
    return var_list

def optimize_iou2(nn_last_layer, correct_label, learning_rate, num_classes):
    logits = tf.reshape(nn_last_layer, (-1, num_classes))    
    labels = tf.reshape(correct_label, (-1, num_classes))    
    '''
    Eq. (1) The intersection part - tf.mul is element-wise, 
    if logits were also binary then tf.reduce_sum would be like a bitcount here.
    '''
    inter=tf.reduce_sum(tf.multiply(logits,labels))
    
    '''
    Eq. (2) The union part - element-wise sum and multiplication, then vector sum
    '''
    union=tf.reduce_sum(tf.subtract(tf.add(logits,labels),tf.multiply(logits,labels)))
    
    # Eq. (4)
    loss=tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.divide(inter,union))
    
    global_step = tf.Variable(0, name='global_step', trainable=False) #learning rate
    train_op=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,var_list=get_trainable_variables(), global_step=global_step)
    return logits, train_op, loss

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
 
    logits = tf.reshape(nn_last_layer, (-1, num_classes))    
    labels = tf.reshape(correct_label, (-1, num_classes))    
   
    # Cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
#    tf.summary.scalar('cross_entropy', cross_entropy)
    
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
    
    # Training loss
    loss = tf.reduce_mean(cross_entropy_loss)
    tf.summary.scalar('loss', loss)

    global_step = tf.Variable(0, name='global_step', trainable=False) #learning rate
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
       
#    grads = optimizer.compute_gradients(loss, var_list=get_trainable_variables())
    grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
    train_op = optimizer.apply_gradients(grads)
    
    merged = tf.summary.merge_all()    
    
    return logits, train_op, cross_entropy_loss, merged

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, merged, input_image,
             correct_label, keep_prob, learning_rate,
             runs_dir=None, data_dir=None, image_shape=None, logits=None):   
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    train_writer = None
    # logger
    if runs_dir is not None:
        if not os.path.exists(runs_dir):
            os.makedirs(runs_dir)
        log_filename = os.path.join(runs_dir, "fcn_training_progress.csv")
        log_fields = ['learning_rate', 'exec_time', 'training_loss']
        log_file = open(log_filename, 'w')
        log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
        log_writer.writeheader()

        summaries_dir = runs_dir + '/summaries'
        if not os.path.exists(summaries_dir):
            os.makedirs(summaries_dir)
            
#        train_writer = tf.summary.FileWriter(summaries_dir, sess.graph)

    totalstarttime = time.clock()

    sess.run(tf.global_variables_initializer())
    
    for i in range(epochs):
                
        training_loss = 0
        training_samples = 0
        print("running epochs:", i)

        # periodically save every 10 epoch runs
        if data_dir is not None and i > 0 and (i % 10) == 0:
            tf.train.Saver().save(sess, 'trained_model',global_step=i)   
            # Save inference data using save_inference_samples
            vgg_helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # start epoch training timer
        starttime = time.clock()

        # train on batches         
        for X, y in get_batches_fn(batch_size):
            training_samples += len(X)            
            
            loss, _, summary = sess.run(
                [cross_entropy_loss, train_op, merged],
                feed_dict={input_image: X, correct_label: y, keep_prob: 0.8})            
            training_loss += loss
        if train_writer is not None:           
            train_writer.add_summary(summary, i)
            
        # calculate training loss
        training_loss /= training_samples
        endtime = time.clock()
        training_time = endtime-starttime
        print("epoch {} execution took {} seconds,".format(i, training_time) +
              " with training loss: {}".format(training_loss))

        # log if doing real training
        if runs_dir is not None:
            log_writer.writerow({
                'learning_rate': learning_rate,
                'exec_time': training_time,
                'training_loss': training_loss})
            log_file.flush()
    totalendtime = time.clock()
    totaltime = totalendtime - totalstarttime
    print("total execution took {} seconds".format(totaltime))
    
    if runs_dir is not None:
        tf.train.Saver().save(sess, 'trained_model')   

def run():
    num_classes = 2
    image_shape = (256, 256)
    home_dir = '/home/frank/selfdriving/sdc_course/CarND-Capstone'

    vgg_path = os.path.join(home_dir, 'vgg')
    data_dir = os.path.join(home_dir, 'bag_dump_just_traffic_light')
    runs_dir = os.path.join(home_dir, 'runs')

    # training hyper parameters
    epochs = 3
    batch_size = 1
    lr = 0.0001
    learning_rate = tf.constant(lr)
    
    with tf.Session() as sess:
        # Path to vgg model
        # Create function to get batches
        get_batches_fn = vgg_helper.gen_batch_function(data_dir, image_shape)
                
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        vgg_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        layers_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        logits, train_op, cross_entropy_loss, merged = optimize(layers_output, correct_label, learning_rate, num_classes)
        
        #Train NN using the train_nn function
        train_nn(
            sess, epochs, batch_size, get_batches_fn, train_op,
            cross_entropy_loss, merged, vgg_input, correct_label, keep_prob,
            lr, runs_dir, data_dir, image_shape, logits)

        #Save inference data using vgg_helper.save_inference_samples
        vgg_helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, vgg_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
