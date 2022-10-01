import tensorflow as tf
from tensorflow import keras
from snn import SNN
from image_loader import imageDirToDataset
from numpy_loader import npDirToDataset

# image_dir = '../tf-autoencoder/graffiti/'
image_dir = 'datasets/first_images.npy'
label_dir = 'datasets/first_labels.npy'

with tf.device('/GPU:0'):
    # Load the data
    # train_features, train_labels = imageDirToDataset(image_dir)
    train_features, train_labels = npDirToDataset(image_dir, label_dir)

    # Compile and fit the network
    snn = SNN()
    snn.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), metrics=['mse']) 
    snn.fit(train_features, train_labels, epochs=5, batch_size=1)

    # # Sample the Fully Connected layer
    for i in range(5):
        results = snn(train_features[i].reshape(1, 2, 32, 32, 3))
        # print("{0}:\nSample label: {1}".format(i, train_labels[i]))
        # print("Siamese Network Sample Results: ", snn.contrastiveLoss)