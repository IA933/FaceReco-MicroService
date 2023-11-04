from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image, ImageDraw
from io import BytesIO
import torch
from ultralytics import YOLO
from cv2.typing import MatLike
import numpy as np
from face_detection import detected_faces_num, detection_pipeline, get_face
from utils import load_image, resize_image
from schema import Point, Box, Vector, VectorMetadata
from creds import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
import pinecone
from vector_db import index_vector

app = FastAPI()

# Initialize the YOLO model
model = YOLO("models/yolov8-face-detection.pt")

@app.get("/")
def read_root():
    return "Free Palestine!"


# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Initialize Pinecone index
index = pinecone.Index(PINECONE_INDEX_NAME)

# Define the face detection and feature extraction functions
# You should replace this with your actual code

@app.post("/register")
async def register(image: UploadFile, name: str):
    try:
        # Read the uploaded image
        image_bytes = await image.read()

        # Save the image to a temporary file (You can modify this based on your storage solution)
        with open(name + ".jpg", "wb") as f:
            f.write(image_bytes)

        # Detect the face and extract features
        face = detection_pipeline(name + ".jpg")

        # Index the features into Pinecone
        if face is not None:
            index_vector(name, face, name)
            return {"message": "Image registered successfully"}

        return {"error": "Face not detected or features extraction failed"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/detect-face")
async def detect_face(image: UploadFile):
    try:
        # Read the uploaded image
        image_bytes = await image.read()

        # Convert image bytes to a PIL Image
        pil_image = Image.open(BytesIO(image_bytes))

        # Convert PIL Image to an OpenCV image (MatLike)
        img = np.array(pil_image)

        # Detect faces using your YOLO model
        boxes = model(img)[0]
        num_detected_faces = detected_faces_num(boxes)
        if num_detected_faces == 0:
            return {"error": "No faces detected"}
        elif num_detected_faces > 1:
            return {"error": "More than one face detected"}

        box = boxes[0]
        box_data = Box(top_left=Point(x=int(box[0]), y=int(box[1]), bottom_right=Point(x=int(box[2]), y=int(box[3]))))

        # Get the detected face
        face = get_face(img, box_data)

        # Resize the face
        face = resize_image(face, height=500)

        # Annotate the image with the face detection box
        annotated_image = pil_image.copy()
        draw = ImageDraw.Draw(annotated_image)
        draw.rectangle([(box_data.top_left.x, box_data.top_left.y), (box_data.bottom_right.x, box_data.bottom_right.y)], outline="red", width=3)
        
        # Save the annotated image to a BytesIO object
        annotated_image_bytes = BytesIO()
        annotated_image.save(annotated_image_bytes, format="JPEG")
        annotated_image_bytes.seek(0)

        return {
            "box": box_data,
            "face": face,
            "annotated_image": annotated_image_bytes,
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/verify")
def verify():
    pass
