import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14")
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# DEFINE THE MODEL
# We use a model that supports 'conversational' task (chat_completion) on Serverless Inference.
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

# ChatHuggingFace wraps the endpoint to use the chat API, which is supported by providers.
model = ChatHuggingFace(llm=llm)

# define schema
class Book(BaseModel):
    title: str = Field(description="The title of the book")
    author: str = Field(description="The author of the book")
    year: int = Field(description="The year the book was published")

parser = PydanticOutputParser(pydantic_object=Book)

template = PromptTemplate(
    template="Provide the details of the book in the following JSON format:\n{format_instructions}\n\nBook: {book_name}",
    input_variables=["book_name"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | model | parser

print("Invoking chain...")
result = chain.invoke({"book_name": "To Kill a Mockingbird"})
print(result)
