from tkinter import *
from interface.utils import tkPack
from PIL import Image as PIL_Image, ImageTk

# A display for pre-detected images
class FeatureViewer():
    def __init__(self, root):
        self.packer = tkPack()
        self.emitClick = None
        self.updateLabels = None
        self.frame, self.img_label = self._setupFeatureViewer(root)

    # Map the latest features and labels
    def setFeatures(self, new_features, all_labels):
        # Clear the previous image tk.Labels
        self.packer.popToLength(2)

        for i, (label, image) in enumerate(map(lambda feature: (feature.label, feature.boxes[-1].image), new_features.collection)):
            pil_img = PIL_Image.fromarray(image[:,:,::-1])
            imgtk = ImageTk.PhotoImage(image=pil_img)
            next_image_label = Label(self.frame)
            next_image_label.configure(image=imgtk)
            next_image_label.image = imgtk
            clicker = Clicker(label, self.emitClick)
            next_image_label.bind("<Button-1>", clicker.featureClick)
            self.packer.append(next_image_label)
            self.packer.append(Label(self.frame, text=label))
            
        self.updateLabels(all_labels)
        self.packer.packThatThangUp()


    def _setupFeatureViewer(self, root, relief=SUNKEN):
        frame = Frame(root, relief=relief, borderwidth=3, width=200, height=600)
        frame.pack_propagate(0)
        label = self.packer.append(Label(frame, text="Feature Viewer"))
        img_label = self.packer.append(Label(frame))

        self.packer.packThatThangUp()
        return frame, img_label

class Clicker():
    def __init__(self, label, emitter):
        self.label = label
        self.emitter = emitter
    
    def featureClick(self, click):
        self.emitter(self.label)
        