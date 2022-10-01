# Give me a path to some imgz, and I give you a dataset :)
import tensorflow as tf
import numpy as np
import random
import math
import os

def preprocess(image, image_size):
    image = tf.io.read_file(image)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    # image = tf.image.rgb_to_grayscale(image)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.reshape(image, shape=(image_size, image_size, 3))
    return image

def imageDirToDataset(path, image_size=32):
    paths = [os.path.join(path, image) for image in os.listdir(path) if '.png' in image or '.jpg' in image]
    letter_indicies = {}
    for i, path in enumerate(paths):
        try:
            letter_indicies[path[-5:-4]].append(i)
        except KeyError as err:
            letter_indicies[path[-5:-4]] = [i]
        
    combination_depth = 5
    cut_dataset = None
    dataset_size = len(paths)
    if (cut_dataset != None): dataset_size = cut_dataset
    dataset_size = math.ceil(dataset_size * combination_depth)
    raw_images = [preprocess(path, image_size) for path in paths]
    print("Dataset Size: ", dataset_size)
    print("Raw_images Size: ", len(raw_images))

    features = []
    labels = []

    # Load The full (or clipped) dataset
    for i, left_img in enumerate(raw_images):
        left_letter = paths[i][-5:-4]
        for j in range(combination_depth):
            k = random.randint(0, len(raw_images) - 1) if j % 2 == 0 else random.choice(letter_indicies[left_letter])
            right_img = raw_images[k]
            right_letter = paths[k][-5:-4]
            features.append([left_img, right_img])
            labels.append([[0 if left_letter == right_letter else 1]])
        if (cut_dataset != None and i == cut_dataset - 1):
            break

    features = np.full((dataset_size, 2, 32, 32, 3), features, dtype='float32')
    labels = np.full((dataset_size, 1, 1), labels, dtype='float32')
    
    return features, labels