import torch
from vector_db import index_vector

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_tensor = torch.rand(512).to(device)

index_vector("2222", random_tensor, "Random Human")