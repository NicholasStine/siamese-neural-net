from interface.auto_suggest.feature_tracker import getAllFeatures
from interface.utils import tkPack
from tkinter import *
import numpy as np
import random
import math

# Save the current UniqueCollection as a siamese dataset
class DatasetController():
    def __init__(self, root, relief=SUNKEN, borderwidth=3):
        self.packer = tkPack()
        self.frame = Frame(root, relief=relief, borderwidth=borderwidth, width=1000, height=200)
        self.frame.pack_propagate(0)
        self.dataset_name = StringVar(self.frame)
        self._setupDatasetController()
        
    # Setup the form and save buttonogla balogna
    def _setupDatasetController(self):
        self.name_entry = self.packer.append(Entry(self.frame, textvariable=self.dataset_name))
        self.save = self.packer.append(Button(self.frame, text="Save Dataset", command=self.saveDataset))
        self.packer.packThatThangUp()
    
    def saveDataset(self):
        # print("DatasetController.saveDataset: ", self.dataset_name.get())
        raw_dataset = getAllFeatures()
        print("raw_dataset: ", raw_dataset)
        images = []
        labels = []
        # Combine each unique feature with a random sample of all other unique features
        for unique in raw_dataset.collection:
            for i, feature in enumerate(unique.boxes):
                pool_samples = random.sample(raw_dataset.collection, min(5, len(raw_dataset.collection)))
                for pool_feature in pool_samples:
                    # print("unique.label: ", unique.label)
                    # print("pool_feature.label: ", pool_feature.label)
                    label = 0 if unique.label == pool_feature.label else 1
                    for unique_box in random.sample(unique.boxes, min(5, len(unique.boxes))):
                        for pool_box in random.sample(pool_feature.boxes, min(5, len(pool_feature.boxes))):
                            labels.append(label)
                            unique_image = unique_box.image
                            pool_image = pool_box.image
                            unique_max_dim = max(unique_image.shape[0], unique_image.shape[1])
                            unique_pad = [unique_max_dim - unique_image.shape[0], unique_max_dim - unique_image.shape[1]]
                            unique_pad = ((math.floor(unique_pad[0] / 2), math.ceil(unique_pad[0] / 2)), (math.floor(unique_pad[1] / 2), math.ceil(unique_pad[1] / 2)), (0, 0))
                            unique_image = np.pad(unique_image, unique_pad)
                            unique_image = np.resize(unique_image, (32, 32, 3))
                            pool_max_dim = max(pool_image.shape[0], pool_image.shape[1])
                            pool_pad = [pool_max_dim - pool_image.shape[0], pool_max_dim - pool_image.shape[1]]
                            pool_pad = ((math.floor(pool_pad[0] / 2), math.ceil(pool_pad[0] / 2)), (math.floor(pool_pad[1] / 2), math.ceil(pool_pad[1] / 2)), (0, 0))
                            pool_image = np.pad(pool_image, pool_pad)
                            pool_image = np.resize(pool_image, (32, 32, 3))

                            images.append([unique_image, pool_image])
                    
        # print("Same Length?\nlen(images): {0}\nlen(labels): {1}".format(len(images), len(labels)))
        # print(labels)
        # print("Big office party I need to go to: ", images[0][0].shape)
        # print("Prepare for every possible disaster: ", images[0][1].shape)
        np.save("datasets/{0}_images".format(self.dataset_name.get()), np.array(images, dtype='float32'), allow_pickle=False)
        np.save("datasets/{0}_labels".format(self.dataset_name.get()), np.array(labels, dtype='float32'), allow_pickle=False)


    def _preprocessImages(self, images):
        pass

    def _preprocessLabels(self, label):
        pass
