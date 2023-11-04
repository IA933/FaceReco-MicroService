from schema import Box
import cv2
from cv2.typing import MatLike

def load_image(img_path: str) -> MatLike:
    return cv2.imread(img_path)

def save_image(img: MatLike, output_path: str):
    cv2.imwrite(output_path, img)
    
def draw_save_detection(img_path: str, output_path: str, box: Box, color=(49, 43, 228), thickness=2):
    img = load_image(img_path)
    cv2.rectangle(img, (box.top_left.x, box.top_left.y), (box.bottom_right.x, box.bottom_right.y), color, thickness)
    cv2.imwrite(output_path, img)
    
def resize_image(img: MatLike, height: float) -> MatLike:
        new_height = height
        aspect_ratio = img.shape[1] / img.shape[0]
        new_width = int(aspect_ratio * new_height)
        resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_image