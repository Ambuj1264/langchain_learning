from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated,Optional,Literal
from pydantic import BaseModel, Field


load_dotenv()

model = ChatOpenAI()

# schema 
class Review(BaseModel):
   key_themes : list[str] = Field(description="Key themes of the review")
   summary :str = Field(description="Summary of the review")
   sentiment: Literal["positive", "negative", "neutral"] = Field(description="Sentiment of the review")
   pros:Optional[list[str]] = Field(default = None, description="best value phone ")
   cons: Optional[list[str]] = Field(default = None, description="Write all the cons of the review")
   rating: Optional[int] = Field(default = None, description="Write the rating of the review")
   name :Optional[str]= Field(default = None, description="Write the name of the review")


structureModel = model.with_structured_output(Review)
result = structureModel.invoke("I recently upgrade to the samsung galaxy s23 ultra,  and i am very satisfied with it. it is the best phone i have ever used.  i have been using it for past 2 months and i am very satisfied with it.  Review by ambuj singh")
print(result.model_dump_json())

