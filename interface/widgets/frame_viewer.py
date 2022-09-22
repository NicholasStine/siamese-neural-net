# Frame Viewer, a frame by frame
# video player with yolo prepredictions

from yolo.yolov3.utils import detect_image, Load_Yolo_model
from yolo.yolov3.configs import *
from interface.utils import tkPack
from interface.auto_suggest.feature_tracker import compareFrames
from interface.framer_cramer.frames import LinkedFrames
from PIL import Image as PIL_Image, ImageTk
from tkinter import filedialog as fd
from tkinter.ttk import *
from tkinter import *
import cv2

# Frame By Frame video player with yolo pre-detection
class FrameViewer():
    def __init__(self, root, skip=1):
        self.skip = skip
        self.features = []
        self.detections = []
        self.packer = tkPack()
        self.updateFeatures = None
        self.frames = LinkedFrames()
        self.cached_detections = None
        self.yolo = Load_Yolo_model()
        self.vid = cv2.VideoCapture('')
        self.frame, self.img_label = self._setupFrameViewer(root)

    # Init and pack all widgetss
    def _setupFrameViewer(self, root, relief=SUNKEN):
        # Init the Frame Viewer... frame lol
        frame = Frame(root, relief=relief, borderwidth=3, width=1000, height=600)
        frame.pack_propagate(0)

        # Add the Labels and Buttons to a packable list
        img_label = self.packer.append(Label(frame))
        label = self.packer.append(Label(frame, text="Frame Viewer"))
        vid_frame = self.packer.append(Frame(frame, width=1000, height=60), side=BOTTOM)
        vid_frame.pack_propagate(0)
        vid_select_button = self.packer.append(Button(vid_frame, text="Select a Video", command=self.selectVideo), side=BOTTOM)
        # placeholder_button = self.packer.append(Button(vid_frame, text="MY GRANDFATHER HENRY FORD"), side=TOP)
        next_frame_button = self.packer.append(Button(vid_frame, text="->", command=self.nextFrame), side=RIGHT)
        previous_frame_button = self.packer.append(Button(vid_frame, text="<-", command=self.prevFrame), side=LEFT)

        # Pack it up!
        self.packer.packThatThangUp()
        
        # Return the frame and current video-frame label
        return frame, img_label

    # Select a video file
    def selectVideo(self):
        file_selector = fd.askopenfilename(title="Select a Video", initialdir='C:/Users/nicks/Code/tf-siamese/yolo/IMAGES')
        self.vid = cv2.VideoCapture(file_selector)
        self.nextFrame(first=True)

    # Go back in the Frames LinkedList
    def prevFrame(self):
        previous_frame = self.frames.prevFrame()
        pil_img = PIL_Image.fromarray(previous_frame)
        pil_width, pil_height = pil_img.size
        pil_img = pil_img.resize((900, int(900 * (pil_height / pil_width))))
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.img_label.configure(image=imgtk)
        self.img_label.image = imgtk

    # Get the next video frame
    def nextFrame(self, first=False):
        # Skip forward by 1 or more frames
        if (not first):
            for i in range(self.skip):
                _, next_frame = self.vid.read()
        else:       
            _, next_frame = self.vid.read()

        # Use cached next frame if available
        if (self.frames.hasNext()):
            next_image = self.frames.nextFrame()
        else:
            # Get the YOLO detected PIL image
            detection, features, box_detections = detect_image(self.yolo, next_frame, '', input_size=YOLO_INPUT_SIZE, rectangle_colors=(255,0,255))
            # Update and cache auto-suggested feature ID's
            self.cached_detections, all_labels = compareFrames(zip(box_detections, features), self.cached_detections)
            self.updateFeatures(self.cached_detections, all_labels)
            next_image = detection[:,:,::-1]
            self.frames.append(next_image)
            self.frames.nextFrame() # Step frame list forward to keep in sync

        # Update the tk.Label where the
        # detected frame is displayed
        pil_img = PIL_Image.fromarray(next_image)
        pil_width, pil_height = pil_img.size
        pil_img = pil_img.resize((900, int(900 * (pil_height / pil_width))))
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.img_label.configure(image=imgtk)
        self.img_label.image = imgtk
        
    # Apply feature combination changes to the cached features
    def updateCached(self, cached, keep, remove):
        cached.pop(remove)
        cached.extend([keep])

    # Getter for the previous frame's detections
    def getCached(self):
        return self.cached_detections