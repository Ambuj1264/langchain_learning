
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=10)

result = llm.invoke("Hello, how are you?")
print(result.content)
