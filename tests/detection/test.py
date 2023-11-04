from face_detection import detect_face, process_image, get_face

img_path = "tests/detection/picture.jpeg"
detection_output_path = "tests/detection/detection.jpeg"
face_output_path = "tests/detection/face.jpeg"

box = detect_face(img_path)
process_image(img_path, detection_output_path, box)
get_face(img_path, face_output_path, box)