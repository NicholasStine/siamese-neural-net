from yolo.yolov3.utils import detect_image, Load_Yolo_model
from yolo.yolov3.configs import *

image_path   = "yolo/IMAGES/hawgs_2.png"
video_path   = "yolo/IMAGES/hog_clip_0.mp4"

yolo = Load_Yolo_model()
detect_image(yolo, image_path, "./IMAGES/hawg_pred_", input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255,0,255))