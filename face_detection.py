import torch
from ultralytics import YOLO
import cv2

from schema import Point, Box

model = YOLO("models/yolov8-face-detection.pt")

def detected_faces_num(tensor: torch.Tensor):
    return tensor.shape[0]

def detect_face(img_path: str):
    boxes = model(img_path)[0]
    
    if detected_faces_num(boxes) == 0:
        raise Exception("No faces detected")
    elif detected_faces_num(boxes) > 1:
        raise Exception("More than one face detected")
    
    box = boxes[0]
    return Box(
        top_left= Point(x=int(box[0]), y=int(box[1])),
        bottom_right= Point(x=int(box[2]), y=int(box[3]))
        )
    
def process_image(img_path: str, output_path: str, box: Box, color=(49, 43, 228), thickness=2):
    img = cv2.imread(img_path)
    cv2.rectangle(img, (box.top_left.x, box.top_left.y), (box.bottom_right.x, box.bottom_right.y), color, thickness)
    cv2.imwrite(output_path, img)
    
def get_face(img_path: str, output_path: str, box: Box):
    img = cv2.imread(img_path)
    cropped_img = img[box.top_left.y:box.bottom_right.y, box.top_left.x:box.bottom_right.x]
    cv2.imwrite(output_path, cropped_img)