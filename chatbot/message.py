from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, max_tokens=100)

messages= [
    SystemMessage(content= "you are helpfull assistant"),
    HumanMessage(content="tell me about the langchain for framework development")
]

result = model.invoke(messages)
# print(result.content)

messages.append(AIMessage(content=result.content))



print(json.dumps([msg.model_dump() for msg in messages], indent=2))