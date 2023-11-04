from face_recognition import embed_img

img_path = "tests/pictures/face.jpeg"

embedding = embed_img(img_path).view(-1)

print(embedding)