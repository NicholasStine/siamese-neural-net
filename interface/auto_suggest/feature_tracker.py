# Tracking features across video frames
# to generate label auto-suggestions
from interface.auto_suggest.features import UniqueFeature, UniqueCollection
all_features = UniqueCollection()
import numpy as np

# Compare the current frame's bounding box
# detections to the previous frame's detections
def compareFrames(current_frames, previous_frames, threshold=20):
    # Break if no previous frames
    if (previous_frames == None): return UniqueCollection(), []

    # For each current frame, check the previous
    # detections within a overlap% threshold
    unique_features = UniqueCollection()
    for box, image in current_frames:
        overlap_percent = 0.0
        overlap_feature = None

        for i, unique in enumerate(previous_frames.collection):
            cached_box = unique.boxes[-1].bbox

            # Use an average of the previous 2-3 boxes to calculate overlap
            if (len(unique.boxes) > 1):
                cached_box = np.array(cached_box)
                cached_box = np.append([cached_box], [unique.boxes[-2].bbox], axis=0)
                if (len(unique.boxes) > 2): cached_box = np.append(cached_box, [unique.boxes[-3].bbox], axis=0)
                cached_box = np.average(cached_box, axis=0).astype(int)
                cached_box = cached_box.tolist()

            # Get and store the highest overlap
            current_overlap = _calculateOverlap(box, cached_box)
            if current_overlap > overlap_percent:
                overlap_percent = current_overlap
                overlap_feature = unique

        # Append to the old unique_feature's cropped_detections
        if (overlap_percent > threshold):
            overlap_feature.append(box, image)
            unique_features.append(overlap_feature)
        
        # Create a new unique feature
        else:
            unique_features.append(UniqueFeature(box, image))

    # Cache and return suggestions
    if (len(unique_features.collection) > 0): all_features.extend(unique_features.collection)
    return unique_features, list(map(lambda feature: feature.label, all_features.collection))
  
# Combine a user selected feature
# with a false negative feature
def combineFeatures(keep_feature, dissolve_feature):
    keeper = all_features[keep_feature]
    remover = all_features[dissolve_feature]
    removed = keeper.absorbFeature(remover)
    if (len(removed.boxes) == 0): all_features.pop(removed)
    return keeper, remover

# Calculate overlap between two bounding boxes
def _calculateOverlap(box_a, box_b):
    try:
        # Break out each box's coordinates
        (a_x1, a_y1), (a_x2, a_y2) = (box_a[0], box_a[1]), (box_a[2], box_a[3])
        (b_x1, b_y1), (b_x2, b_y2) = (box_b[0], box_b[1]), (box_b[2], box_b[3])

        # Calculate the current box and intersection box areas
        a_area = (a_x2 - a_x1) * (a_y2 - a_y1)
        dx = min(a_x2, b_x2) - max(a_x1, b_x1)
        dy = min(a_y2, b_y2) - max(a_y1, b_y1)

        # Return overlap percentage
        if (dx >= 0) and (dy >= 0):
            return (dx * dy) / a_area * 100
        else:
            return 0.00
    except IndexError as err:
        return 0.00