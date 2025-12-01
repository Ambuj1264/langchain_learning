
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1.8, max_tokens=100)


result = model.invoke("suggest me for indian state names")
print(result.content)
