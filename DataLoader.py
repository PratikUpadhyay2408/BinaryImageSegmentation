import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

"""

"""


class DataLoader:
    """
    This class provides functionality to load, augment and convert image data into Tensorflow datasets
    """
    root_img_dir = None
    root_mask_dir = None
    width = None
    height = None
    batch_size = None
    shuffle_buffer_size = None

    def __init__( self, root_img_dir='', root_mask_dir='', width=448 , height=448, batch_size=4 , shuffle_buffer_size=8 ):
        assert root_img_dir
        assert root_mask_dir

        self.root_img_dir = root_img_dir
        self.root_mask_dir = root_mask_dir
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size

    def make_image(self, img_path, label_path):
        image = tf.io.read_file(img_path)
        image = tf.io.decode_jpeg(image)
        image = tf.image.resize(image, [self.height, self.width])
        image = image/255

        label = tf.io.read_file(label_path)
        label = tf.io.decode_jpeg(label)
        label = tf.image.resize(label, [self.height, self.width])
        label = label/255

        return image, label

    def makedataset(self):
        img_list = sorted([os.path.join(self.root_img_dir,i) for i in os.listdir(self.root_img_dir)])
        mask_list = sorted([os.path.join(self.root_mask_dir,i) for i in os.listdir(self.root_mask_dir)])

        dataset = tf.data.Dataset.from_tensor_slices( ( np.array(img_list), np.array(mask_list) ) )
        dataset = dataset.map(self.make_image)
        dataset = dataset.shuffle(self.shuffle_buffer_size).batch(self.batch_size)

        return dataset


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow((display_list[i]))
        plt.axis('off')
    plt.show()