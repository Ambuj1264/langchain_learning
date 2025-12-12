from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template= "generate  instresting fact about {topic}",
    input_variables=["topic"],
)

prompt1 = PromptTemplate(
    template= "generate a 2 point summary about {text}",
    input_variables=["text"],
)

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

parser = StrOutputParser()

chain = prompt | model | parser | prompt1 | model | parser

print(chain.invoke({"topic": "AI"}))
