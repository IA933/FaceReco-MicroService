from PIL import Image
import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

def embed_img(img_path: str):
    img = Image.open(img_path)
    input_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return resnet(input_tensor)