# A siamise network with a GUI
*(Powered by tensorflow)* :mechanical_arm: :brain:

## GUI for Building Datasets

### Label Auto-Suggest
Uses a pre-built YOLO network to accelerate
cropping and structuring videos into siamese
labeled datasets (0 for same, 1 for different).

The YOLO bounding-box output is fed into an
aproximate feature tracking algorithm. This
produces auto-suggested labels to reduce the 
dreaded task of manual labeling.

### Controls
This is built into a tkinter GUI for processing
video files frame by frame. The user can:

## The Siamese Network
