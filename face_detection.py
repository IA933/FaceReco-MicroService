import torch
from ultralytics import YOLO
from cv2.typing import MatLike

from utils import load_image, resize_image
from schema import Point, Box

model = YOLO("models/yolov8-face-detection.pt")

def detected_faces_num(tensor: torch.Tensor):
    return tensor.shape[0]

def detect_face(img_path: str) -> Box:
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
    
def get_face(img: MatLike, box: Box):
    face = img[box.top_left.y:box.bottom_right.y, box.top_left.x:box.bottom_right.x]
    return face

def detection_pipeline(img_path: str):
    img = load_image(img_path)
    box = detect_face(img_path)
    face = get_face(img, box)
    face = resize_image(face, height=500)
    return box, face