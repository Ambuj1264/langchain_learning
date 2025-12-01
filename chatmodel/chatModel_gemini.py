from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

result = model.invoke("suggest me for indian state names")
print(result.content)

