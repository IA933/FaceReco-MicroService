from face_recognition import embed_img

img_path = "tests/pictures/face.jpeg"

embedding = embed_img(img_path)

print(embedding.shape)