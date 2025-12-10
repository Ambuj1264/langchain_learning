from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="TinyLLama/TinyLLama-1.1B-Chat-v1.0",
    task = "text-generation"

)
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of France?")
print(result.content)