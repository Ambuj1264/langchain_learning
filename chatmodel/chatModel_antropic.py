from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv
load_dotenv()

model = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.5)

result = model.invoke("suggest me for indian state names")  
print(result.content)


