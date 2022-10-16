import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

TRAIN_IMG_DIR = r'G:\AI\BinaryImageSegmentation\Data\train\images'
TRAIN_MASK_DIR = r'G:\AI\BinaryImageSegmentation\Data\train\masks'


def make_image(x, y):
    image = tf.io.read_file(x)
    image = tf.image.decode_image(image, channels=3)

    image = image/255

    label = tf.io.read_file(y)
    label = tf.image.decode_image(label)
    label = label/255

    return image,label


def display(display_list):
    plt.figure(figsize=(15,15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow( display_list[i] )
        plt.axis('off')
    plt.show()


train_img = sorted([os.path.join(TRAIN_IMG_DIR,i) for i in os.listdir(TRAIN_IMG_DIR)])
train_mask = sorted([os.path.join(TRAIN_MASK_DIR,i) for i in os.listdir(TRAIN_MASK_DIR)])
train_data = tf.data.Dataset.from_tensor_slices( (np.array(train_img), np.array(train_mask)) )
train_data = train_data.map(make_image)

for image, mask in train_data.take(1):
    display([image, mask])