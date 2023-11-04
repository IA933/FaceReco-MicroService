import torch
from vector_db import query_vector

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_tensor = torch.rand(512).to(device)

result = query_vector(random_tensor)
print(result)