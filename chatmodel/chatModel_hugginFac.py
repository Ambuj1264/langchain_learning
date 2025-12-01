from langchain_huggingface  import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is 2+2?")

print(result.content)