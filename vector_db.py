from config import threshold
from creds import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
from schema import Vector, VectorMetadata

import torch
import pinecone      

pinecone.init(      
	api_key=PINECONE_API_KEY,      
	environment=PINECONE_ENVIRONMENT      
)

index = pinecone.Index(PINECONE_INDEX_NAME)

def index_vector(id: str, vector: torch.Tensor, name: str):
	vector = Vector(id=id, values=vector.tolist(), metadata=VectorMetadata(name=name))
	index.upsert([vector.model_dump()])
 
def query_vector(vector: torch.Tensor, top_k: int = 1):
	result = index.query(
		vector=vector.tolist(),
		top_k=top_k,
		include_distances=True,
		include_metadata=True
	)
 
	if len(result["matches"]) == 0:
		return None

	if result["matches"][0]["score"] > threshold:
		return None

	return result["matches"][0]