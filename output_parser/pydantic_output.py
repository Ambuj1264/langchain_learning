import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14")

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# We switch to OpenAI because small open-source models (like Gemma-2b)
# struggle with the complex JSON schemas required by PydanticOutputParser.
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
result = chain.invoke({"book_name": "To Kill a Mockingbird"})
print(result)