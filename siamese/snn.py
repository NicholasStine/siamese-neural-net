# Attempt 1 at a siamese network
# This is a port of a pytorch implementation to tensorflow:
# https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942

# Performance:
# The network consistently stalled at a contrastive loss
# of around 0.70 - 0.75 (sigmoid)
# I think it's failing for lack of pre-training

# Imports
from tensorflow.math import add as tfAdd, pow as tfPow, sqrt as tfSqrt, minimum as tfMin, reduce_sum as tfReduceSum, sigmoid as tfSigmoid
from tensorflow import ones as Ones, constant as Constant, split as Split, squeeze as Squeeze, expand_dims as ExpandDims, GradientTape
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, LayerNormalization, Flatten
from tensorflow.keras import Sequential, Model, layers, models, losses
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset
import tensorflow.keras.backend as K
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import math
import os

# Define the Siamese Network
class SNN(Model):
    def __init__(self, margin=1.0):
        super().__init__()
        # Load Pre-trained network and lock weights
        # resnet = tf.keras.applications.densenet.DenseNet121(include_top=False, input_shape=(32, 32, 3))
        # resnet = tf.keras.applications.resnet.ResNet101(include_top=False, input_shape=(32, 32, 3))
        # resnet = tf.keras.applications.efficientnet_v2.EfficientNetV2B3(include_top=False, input_shape=(32, 32, 3))
        # for layer in resnet.layers:
        #     layer.trainable = False

        # # Flatten and reduce the output to 10 sigmoid activated output nodes
        # x = layers.Flatten()(resnet.output)
        # x = layers.Dense(128, activation='relu')(x)
        # predictions = layers.Dense(64, activation='softmax')(x)

        # self.cnn = Model(resnet.input, predictions)
        self.margin = margin
        self.cnn = Sequential(name="CNN", layers=[
            # Input and first feature convolution
            Input((32, 32, 3)),
            Conv2D(64, (2, 2), activation="relu"),
            LayerNormalization(),
            MaxPool2D(2),

            # Second convolution w/ dropout noise
            Conv2D(128, (2, 2), activation='relu'),
            LayerNormalization(),
            MaxPool2D(2),
            Dropout(0.3),

            # Third convolution w/ dropout noise
            # Conv2D(256, (2, 2)),
            Conv2D(256, (2, 2), activation='relu'),
            LayerNormalization(),
            MaxPool2D(2),
            Dropout(0.3),

            # Conv2D(512, (2, 2)),
            Conv2D(512, (2, 2), activation='relu'),
            MaxPool2D(2),
            Dropout(0.3),
            
            # Flatten the data into a dense layer
            Flatten(),
            Dense(128, activation='relu'),
            Dense(64, activation='softmax'),
        ])

    # The forward pass method
    def call(self, images):
        image_a, image_b = Split(images, 2, axis=1)
        image_a = Squeeze(image_a, axis=1)
        image_b = Squeeze(image_b, axis=1)

        # Run each image through the CNN and Fully Connnected Networks
        result_a = self.cnn(image_a)
        result_b = self.cnn(image_b)

        # Return results a and b
        return result_a, result_b

    # Euclidian Loss function 
    # Dw = (a^2 + b^2)^-2 <-- Note the negative for square-root ;)
    def euclidianLoss(self, a, b):
        return tf.math.reduce_euclidean_norm(tf.concat([a, b], -1))
        # return tf.math.sqrt(tf.math.square(a) + tf.math.square(b))

    # The contrastive loss function
    # (1 - Y) * 1/2 * (Dw)^2 + (Y) * 1/2 * (max(0, margin - Dw))^2
    def contrastiveLoss(self, result_a, result_b, label):
        # Calculate Euclidian Loss
        Dw = self.euclidianLoss(result_a, result_b)

        # Calculate Term 1
        t1 = 1 - label
        t1 = t1 * 0.5
        t1 = t1 * tf.math.sqrt(Dw)
        
        # Calculate Term 2
        t2 = label * 0.5
        t2 = t2 * tf.math.pow(tf.math.maximum(0.0, self.margin - Dw), 2)

        # Return the combined terms
        return t1 + t2

    # The training step
    # This is where the call and contrastiveLoss methods come together
    def train_step(self, data):
        # Unpack the data for the current training step
        images, label = data

        with GradientTape() as tape:
            # Feed the current data item through the network
            result_a, result_b = self(images)
            
            # Get the loss from the encoder results
            loss = self.contrastiveLoss(result_a, result_b, label)

        # Combine the custom loss results with GradientTape
        # to calculate the new gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update the model
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(label, result_a)

        return { 'loss': loss, **{m.name: m.result() for m in self.metrics} }


# Load just one or a few images for testing
# train_labels.append([1.0])
# train_labels.append([0.0])
# train_images.append([raw_images[1], raw_images[8]])
# train_images.append([raw_images[1], raw_images[1]])
# train_data = np.full((2, 2, 32, 32, 3), train_images, dtype='float32') # Use when testing 1 image at a time

# Show training image(s)
# for i in range(111, 115):
#     train_image = Image.fromarray(np.uint8(train_data[i][0] * 255))
#     base_title = "{0} : ".format("same" if train_labels[i][0][0] == 1.0 else "different")
#     train_image.show(title="{0}{1}".format(base_title, '(-)'))
#     train_image = Image.fromarray(np.uint8(train_data[i][1] * 255))
#     train_image.show(title="{0}{1}".format(base_title, '(+)'))



    # Testing the same image as a left and right 
    # image pair to verify loss function output
    # (result_a_0, result_b_0) = snn(train_data[0].reshape(1, 2, 32, 32, 3))
    # (result_a_1, result_b_1) = snn(train_data[1].reshape(1, 2, 32, 32, 3))

    # Link training to Tensor Board
