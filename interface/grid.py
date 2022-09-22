from tkinter import *
from interface.widgets.frame_viewer import *
from interface.widgets.feature_viewer import *
from interface.widgets.class_selector import *

def drawGrid(root):
    # Init any class-based widgets
    frame_viewer = FrameViewer(root, skip=10)
    feature_viewer = FeatureViewer(root)
    class_selector = ClassSelector(root)

    # A confrence of interwidgitinental setters and getters
    frame_viewer.updateFeatures = feature_viewer.setFeatures
    feature_viewer.emitClick = class_selector.featureClick
    feature_viewer.updateLabels = class_selector.setLabels
    class_selector.updateFeatures = frame_viewer.updateCached
    class_selector.getCached = frame_viewer.getCached

    # Draw all modules
    vid_frame = frame_viewer.frame
    cropped_frame = feature_viewer.frame
    control_frame = class_selector.frame
    
    # Organize the modules in a grid
    vid_frame.grid(row=0, column=0, columnspan=2, rowspan=2)
    cropped_frame.grid(row=0, column=2, columnspan=1, rowspan=2)
    control_frame.grid(row=2, column=0, columnspan=3, rowspan=1, ipadx=450, ipady=50, padx=20, pady=20)
