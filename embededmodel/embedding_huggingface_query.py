
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# query = "What is the capital of France?"
document = ["What is the capital of France?", "What is the capital of India?",
    "What is the capital of Australia?" ,
    "What is the capital of China?",
    "What is the capital of Japan?"
]

query_embedding = embeddings.embed_documents(document)

print(str(query_embedding))
