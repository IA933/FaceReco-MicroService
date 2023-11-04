from creds import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME

import pinecone      

pinecone.init(      
	api_key=PINECONE_API_KEY,      
	environment=PINECONE_ENVIRONMENT      
)      
index = pinecone.Index(PINECONE_INDEX_NAME)