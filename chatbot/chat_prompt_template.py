from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# chat_template = ChatPromptTemplate.from_messages([
#     SystemMessage(content="You are a helpful {domain} expert"),
#     HumanMessage(content="explain in simple terms what is {user_input}")
# ])

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'explain in simple terms what is {user_input}')
])

prompt = chat_template.invoke({
    "domain": "cricket expert",
    "user_input": "who is current best player in the world"
})

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, max_tokens=100)

result = model.invoke(prompt)
newQuery = model.invoke("what is your gpt version")
print(result.content)
print(newQuery.content)