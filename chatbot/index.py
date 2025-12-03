from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=100)

# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello, how are you?"}
# ]

# result = model.invoke(messages)
# print(result.content)
chat_history=[]

while True:
    user_input = input("User: ")
    chat_history.append({"role": "user", "content": user_input})
    if user_input== 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append({"role": "assistant", "content": result.content})
    # print("Chat History: ", chat_history)
    print("Assistant: ", result.content)
