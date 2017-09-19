import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
import yaml
from glob import glob
#from urllib.request import urlretrieve
#from tqdm import tqdm


#class DLProgress(tqdm):
#    last_block = 0

#    def hook(self, block_num=1, block_size=1, total_size=None):
#        self.total = total_size
#        self.update((block_num - self.last_block) * block_size)
#        self.last_block = block_num


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    print('data_folder:'+data_folder)

    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        gt_image_data = []
        labels = {'Red':0, 'Yellow':1,'Green':2,'Unknown':4}
# Read YAML file
        with open(os.path.join(data_folder,"sloth_capstone_real_data.yaml"), 'r') as stream:
            gt_data = yaml.load(stream)
            for o in gt_data:
                a = o['annotations']
                name = o['filename']
                label = a[0]['class']
                x = int(a[0]['x']) 
                y = int(a[0]['y'])
                width = int(a[0]['width'])
                height = int(a[0]['height']) 
                gt_image_data.append((x,y,width,height,name, label))   
        
        background_color = np.array([255, 0, 0])

        random.shuffle(gt_image_data)
        for batch_i in range(0, len(gt_image_data), batch_size):
            images = []
            gt_images = []
            for gt in gt_image_data[batch_i:batch_i+batch_size]:
                (x,y,width,height,name, label) = gt
                image_file = os.path.join(data_folder,name)

                image = scipy.misc.imread(image_file)
                #(1096, 1368, 3)
                orig_shape = image.shape 
                image = scipy.misc.imresize(image, image_shape)
                
                gt_image = np.full(orig_shape, True, dtype=bool)
                
                gt_image[y:y+height,x:x+width,] = False#only light, no light at the beginninglabels[label]
                gt_image = scipy.misc.imresize(gt_image, image_shape)
                #shape (256, 512)
                gt_bg = gt_image[:,:,0]
                #shape (256, 512, 1)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                #shape (256, 512, 2)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                
                images.append(image)
                gt_images.append(gt_image)
                
                #flip images to double dataset
                images.append(np.fliplr(image))
                gt_images.append(np.fliplr(gt_image))

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'test', '*.jpg')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, data_dir, image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

def test_yaml():
# Read YAML file
    with open("/home/frank/selfdriving/sdc_course/CarND-Capstone/bag_dump_just_traffic_light/sloth_capstone_real_data.yaml", 'r') as stream:
        gt_data = yaml.load(stream)
        
        image_shape = (512,512)
        labels = {'Red':0, 'Yellow':1,'Green':2,'Unknown':4}
        
        for o in gt_data:
                a = o['annotations']
                name = o['filename']
                label = a[0]['class']
                x = int(a[0]['x']) 
                y = int(a[0]['y'])
                width = int(a[0]['width'])
                height = int(a[0]['height']) 
                print(x,y,width,height,name, label)
                
                gt_image_file = os.path.join('/home/frank/selfdriving/sdc_course/CarND-Capstone/bag_dump_just_traffic_light','gt',name)                
                image_file = os.path.join('/home/frank/selfdriving/sdc_course/CarND-Capstone/bag_dump_just_traffic_light',name)
                
                image = scipy.misc.imread(image_file)
                orig_shape = image.shape 
                image = scipy.misc.imresize(image, image_shape)
                
                gt_image = np.zeros(orig_shape)
                
                gt_image[y:y+height,x:x+width,] = 1#only light, no light at the beginninglabels[label]
                gt_image = scipy.misc.imresize(gt_image, image_shape)
                #shape (512, 512)
                gt_bg = gt_image[:,:,0]
                #shape (512, 512, 1)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                #shape (512, 512, 3)
                gt_image = np.concatenate((gt_bg, gt_bg,gt_bg), axis=2)
                 
                scipy.misc.imsave(gt_image_file,gt_image)
               
                   
        
if __name__ == '__main__':
    test_yaml()
