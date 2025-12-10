
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated,Optional,Literal


load_dotenv()

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, max_tokens=100)

#schema 
class Review (TypedDict):
    summery: Annotated[str,"A summary of the review"]
    sentiment:Annotated[str,"The sentiment of the review"]
    rating:Annotated[Optional[int],"The rating of the review"]
    stars:Annotated[Literal[1,70,13],"The stars of the review"]


structureModel = model.with_structured_output(Review)
result = structureModel.invoke("What is the capital of France? 5")
print(result)