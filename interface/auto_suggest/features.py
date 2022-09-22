# Just your average indebted-
# servitude towards the OOP empire :)

# Real Talk: It's an identifiable feature
# that can be tracked and labeled across frames

from interface.auto_suggest.labels import MASTER_LABEL_LIST
import random

# A cropped image / bounding box container class
class Cropped():
    def __init__(self, bbox, image):
        self.bbox = bbox
        self.image = image

    def __str__(self):
        return str(self.bbox)

# An ID-able feature with a collection of frames
class UniqueFeature():
    # Static list of already-taken labels
    used = []

    def __init__(self, bbox, image):
        self.boxes = [Cropped(bbox, image)]
        self.label = self._label()

    def __str__(self):
        return "{0} ({1} features)".format(self.label, len(self.boxes))

    def _label(self):
        # Get label and ensure it's unique
        for i in range(100): # Limit 100 to avoid endless loop
            label = random.choice(MASTER_LABEL_LIST)
            if (label not in UniqueFeature.used): break
        
        # Cache and return label
        UniqueFeature.used.append(label)
        return label

    def append(self, bbox, image):
        self.boxes.append(Cropped(bbox, image))

    def absorbFeature(self, dissolve_feature):
        self.boxes.append(dissolve_feature.boxes[-1])
        dissolve_feature.boxes.pop()
        return dissolve_feature

# A list of UniqueFeatures
class UniqueCollection():
    def __init__(self):
        self.collection = []

    # TODO: Broken
    def __getitem__(self, label_i):
        item = list(filter(lambda feature: (feature.label == label_i), self.collection))
        return item[0]

    def __str__(self):
        return str(list(map(lambda feature: str(feature), self.collection)))

    def append(self, unique):
        self.collection.append(unique)

    def extend(self, unique_list):
        verified_unique = filter(lambda unique: unique not in self.collection, unique_list)
        self.collection.extend(verified_unique)
    
    def pop(self, removed):
        remove_index = next(i for i, feature in enumerate(self.collection) if feature.label == removed.label)
        self.collection.pop(remove_index)