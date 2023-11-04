from PIL import Image
import torch
from cv2.typing import MatLike
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

def embed_local_image(img_path: str) -> torch.Tensor:
    img = Image.open(img_path)
    input_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return resnet(input_tensor).view(-1)

def embed_img(img: MatLike) -> torch.Tensor:
    input_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return resnet(input_tensor).view(-1)