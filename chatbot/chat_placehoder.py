from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


# chat template 
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. {domain}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "What is the capital of France? {user_input}")
])


# load chat history 
chat_history = []
with open("chatbot/chat_history.txt") as f:
    chat_history.extend([HumanMessage(content=line.strip()) for line in f.readlines() if line.strip()])

print("===========chat_history", chat_history ,"=========chat history")

# create prompt 
prompt = chat_template.invoke({
    "chat_history": chat_history,
    "domain": "cricket expert",
    "user_input": "what is mine previous query"
})

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, max_tokens=100)

result = model.invoke(prompt)
print(result.content)
