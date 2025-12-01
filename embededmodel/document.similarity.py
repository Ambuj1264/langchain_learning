
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "Delhi is capital of India",
    "karachi is capital of Pakistan",
    "Tokyo is capital of Japan",
    "Beijing is capital of China",
    "Paris is capital of France"
]

query = "What is the capital of India?"

query_embedding = embeddings.embed_query(query)
documents_embedding = embeddings.embed_documents(documents)

similarity_scores = cosine_similarity([query_embedding], documents_embedding)[0]


index, score = sorted(   list(enumerate(similarity_scores)), key=lambda x: x[1])[-1]

print(documents[index])






