from utils import save_image, draw_save_detection
from face_detection import detection_pipeline

dir = "tests/pictures"
img_path = f"{dir}/picture.jpeg"
detection_output_path = f"{dir}/detection.jpeg"
face_output_path = f"{dir}/face.jpeg"

box, face = detection_pipeline(img_path)

draw_save_detection(img_path, detection_output_path, box)
save_image(face, face_output_path)