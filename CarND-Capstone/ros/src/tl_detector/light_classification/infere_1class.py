import tensorflow as tf
import scipy.misc
import numpy as np
from moviepy.editor import *
from helper import gen_test_output

import os.path
import shutil
import time
from glob import glob

saved_mode_path = './data/'
input_video_path = './video/harder_challenge_video.mp4'
input_images_path = './data/data/testing'
output_video_path = './video/'
output_images_path = './runs'
#image_shape = (160, 576)
image_shape = (256, 256)
num_classes = 5

def load_pretrained_model(sess, model_path):
    saver = tf.train.import_meta_graph(model_path + 'model.meta')
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    all_vars = tf.get_collection('vars')

    # sanity check after loading, get some printout to screen
    for var in all_vars:
        var_ = sess.run(var)
        print(var_)

    # get default model graph for current session (on which we have restored the model)
    graph = tf.get_default_graph()
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    input_image = graph.get_tensor_by_name('image_input:0')
    logits = graph.get_tensor_by_name('logits:0')

    return input_image, logits, keep_prob

def segment_frame(sess, frame, keep_prob, input_image, logits):
    frame = scipy.misc.imresize(frame, image_shape)
    feed_dict = {keep_prob: 1.0, input_image: [frame]}

    run_op = tf.nn.softmax(logits)
    frame_softmax = sess.run([run_op], feed_dict=feed_dict)
    frame_softmax = frame_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (frame_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode='RGBA')
    segmented_frame = scipy.misc.toimage(frame)
    segmented_frame.paste(mask, box=None, mask=mask)
    return np.array(segmented_frame)

def segment_video(sess, keep_prob, input_image, logits, input_video_path, output_video_path):
    video_frames = []
    input_video = VideoFileClip(input_video_path)
    for input_frame in input_video.iter_frames():
        output_frame = segment_frame(sess, input_frame, keep_prob, input_image, logits)
        video_frames.append(output_frame)
    output_video = ImageSequenceClip(video_frames, fps=30)
    output_video.write_videofile(output_video_path + 'output.mp4', audio=False)

def segment_images(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, num_classes):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        #sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data/testing'), image_shape, num_classes)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

def run():
    sess = tf.Session()
    in_image, logits, keep_prob = load_pretrained_model(sess, saved_mode_path)
    #segment_video(sess, keep_prob, in_image, logits, input_video_path, output_video_path)
    segment_images(output_images_path, saved_mode_path, sess, image_shape, logits, keep_prob, in_image, num_classes)


if __name__ == '__main__':
    run()
