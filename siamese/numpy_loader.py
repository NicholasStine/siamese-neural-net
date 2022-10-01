# Load an image/label dataset that's already in np form
import numpy as np

def npDirToDataset(image_path, label_path):
    features = np.load(image_path)
    labels = np.load(label_path)

    print("No, no, no more lol: \n{0}\n{1}".format(features, labels))

    return features, labels
