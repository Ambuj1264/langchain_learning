from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template= "generate 5 instresting fact about {topic}",
    input_variables=["topic"],
)

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({"topic": "AI"}))

