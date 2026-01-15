import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.7
)
prompt = PromptTemplate(
    template="Write a summary for the following text {text}",
    input_variables=["text"]
)

parser= StrOutputParser()
chain = prompt | model | parser

chain.invoke({"text": "Hello World"})

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "movie.txt")

loader=TextLoader(
    file_path
)
documents=loader.load()
print(documents[0].metadata, documents[0].page_content)

print(type(documents))

