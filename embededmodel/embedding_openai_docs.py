from langchain_openai import  OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=45)

document= ["Delhi is capital of India",
            "karachi is capital of Pakistan",
            "Tokyo is capital of Japan",
            "Beijing is capital of China",
            "Paris is capital of France"
    ]

query = "What is the capital of France?"

query_embedding = embeddings.embed_documents(document)

print(str(query_embedding))
