from langchain_openai import  OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=45)

query = "What is the capital of France?"

query_embedding = embeddings.embed_query(query)

print(str(query_embedding))
