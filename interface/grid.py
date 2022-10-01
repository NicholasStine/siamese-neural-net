from tkinter import *
from interface.widgets.frame_viewer import *
from interface.widgets.feature_viewer import *
from interface.widgets.class_selector import *
from interface.widgets.dataset_controls import *

def drawGrid(root):
    # Init any class-based widgets
    frame_viewer = FrameViewer(root, skip=10)
    feature_viewer = FeatureViewer(root)
    class_selector = ClassSelector(root)
    dataset_controls = DatasetController(root)

    # A confrence of interwidgitinental setters and getters
    frame_viewer.updateFeatures = feature_viewer.setFeatures
    feature_viewer.emitClick = class_selector.featureClick
    feature_viewer.updateLabels = class_selector.setLabels
    class_selector.updateFeatures = frame_viewer.updateCached
    class_selector.getCached = frame_viewer.getCached

    # Draw all modules in a grid
    frame_viewer.frame.grid(row=0, column=0, columnspan=2, rowspan=2)
    feature_viewer.frame.grid(row=0, column=2, columnspan=1, rowspan=2)
    class_selector.frame.grid(row=2, column=2, columnspan=1, rowspan=1)
    dataset_controls.frame.grid(row=2, column=0, columnspan=2, rowspan=1)
