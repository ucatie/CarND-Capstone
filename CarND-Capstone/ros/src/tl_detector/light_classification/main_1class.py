import os.path
import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import datetime

from moviepy.editor import *


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
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
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    vgg_input_tensor = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # do 1x1 convolutions of all three layers... this makes sure that everything has the same dimensions so layers can be added for the skips
    layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    # upscale layer 7, add it to layer 4
    total = tf.layers.conv2d_transpose(layer7, num_classes, 4, 2, 'SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    total = tf.add(total, layer4)

    # upscale total, add it to layer 3
    total = tf.layers.conv2d_transpose(total, num_classes, 4, 2, 'SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    total = tf.add(total, layer3)

    # upscale and return
    total = tf.layers.conv2d_transpose(total, num_classes, 16, 8, 'SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    return total
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label, name="entropy"))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
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
    # TODO: Implement function
    tf.global_variables_initializer()

    print("Training starts at {}".format(datetime.datetime.now().time()))

    for epoch_i in range(epochs):
        print("Starting epoch {}.".format(epoch_i + 1))
        batch_counter = 0
        total_loss = []
        for train_img, label in get_batches_fn(batch_size):
            # prepare feed_dict with the data from the current batch
            feed_dict={input_image: train_img, correct_label: label, keep_prob: 0.5, learning_rate: 0.00001}
            # trigger train operation
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict)
            # just for the sake of presenting statistics during the training process
            total_loss.append(loss)
            batch_counter = batch_counter + 1
        # after each batch in completed, print some statistics... maybe this won't work since feed_dict is out of scope
        train_loss = sum(total_loss) / (batch_counter * batch_size)
        print("Epoch completed after {} batches with train loss = {}".format(batch_counter, train_loss))
        # TODO: adapt learning rate as long as the variations in the training loss indicate it would be wise

    print("Training completes at {}".format(datetime.datetime.now().time()))

tests.test_train_nn(train_nn)

def process_frame(image, sess, logits, keep_prob, image_pl):
    result_image = image
    image_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_pl: [image]})
    image_softmax = im_softmax[0][:, 1].reshape(image.shape[0], image.shape[1])
    segmentation = (im_softmax > 0.5).reshape(image.shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    final_image = scipy.misc.toimage(image)
    final_image.paste(mask, box=None, mask=mask)

    return final_image

def run():
    num_classes = 2
    #num_classes = 5
    #image_shape = (160, 576)
    image_shape = (256, 256)
    data_dir = './data'
    runs_dir = './runs'
    input_video_path = './video/project_video.mp4'
    ouput_video_path = './video/project_video_output.mp4'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # force the use of CPU to test, since we need more than 4GB VRAM for this
    #config = tf.ConfigProto(device_count = {'GPU': 0})
    with tf.Session() as sess:
    #with tf.Session(config=config) as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        #get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.int32, shape=[None, None, None, num_classes])
        learn_rate = tf.placeholder(tf.float32)
        input_layer, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        output = layers(layer3, layer4, layer7, num_classes)
        logits, train_op, x_entropy_loss = optimize(output, correct_label, 0.00001, num_classes)
        sess.run(tf.global_variables_initializer())
        # TODO: Train NN using the train_nn function
        train_nn(sess, 50, 15, get_batches_fn, train_op, x_entropy_loss, input_layer, correct_label, keep_prob, learn_rate)

        # also save model
        saver = tf.train.Saver()
        saver.save(sess, 'data/model.ckpt')
        saver.export_meta_graph('data/model.meta')
        tf.train.write_graph(sess.graph_def, "./data/", "model.pb", False)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_layer, num_classes)

        # OPTIONAL: Apply the trained model to a video --- this is done in 'infere.py'
        #output_images = []
        #clip = VideoFileClip(input_video_path)
        #for frame in clip.iter_frames():
        #    processed_image = process_frame(frame, sess, logits, keep_prob, input_layer)
        #    output_images.append(processed_image)
        #output_clip = ImageSequenceClip(output_images, fps=30)
        #output_clip.write_videofile(ouput_video_path, audio=False)

if __name__ == '__main__':
    run()
