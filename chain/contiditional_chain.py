from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# ---------------- Schema ----------------
# Pydantic model to structure the classification output
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the feedback"
    )

# Parser that converts LLM output to Pydantic Feedback object
parser2 = PydanticOutputParser(pydantic_object=Feedback)

# ---------------- Prompts ----------------
# Prompt to classify feedback sentiment
prompt1 = PromptTemplate(
    template="""
    Classify the feedback as positive or negative.
    Feedback: {feedback}
    {format_instructions}
    """,
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": parser2.get_format_instructions()
    },
)

# Prompt for positive response
prompt2 = PromptTemplate(
    template="Write an appropriate positive response for the feedback: {feedback}",
    input_variables=["feedback"],
)

# Prompt for negative response
prompt3 = PromptTemplate(
    template="Write an appropriate negative response for the feedback: {feedback}",
    input_variables=["feedback"],
)

parser = StrOutputParser()

# ---------------- Chains ----------------
# Chain to classify feedback - outputs Pydantic Feedback object with .sentiment attribute
classify_chain = prompt1 | model | parser2

# Branch chain that routes based on sentiment value in the input dict
# Expects input dict with keys: {"feedback": "...", "sentiment": "positive/negative"}
branch_chain = RunnableBranch(
    # If sentiment is positive, generate positive response
    (lambda x: x["sentiment"] == "positive", prompt2 | model | parser),
    # If sentiment is negative, generate negative response
    (lambda x: x["sentiment"] == "negative", prompt3 | model | parser),
    # Default fallback
    RunnableLambda(lambda x: "Could not classify the feedback"),
)

# ---------------- Full Pipeline ----------------
# RunnablePassthrough.assign() preserves original input {"feedback": "..."} and adds "sentiment" key
# by running classify_chain and extracting .sentiment from the Pydantic result
full_chain = RunnablePassthrough.assign(
    sentiment=lambda x: classify_chain.invoke(x).sentiment
) | branch_chain

# ---------------- Run ----------------
result = full_chain.invoke(
    {"feedback": "The food was delicious and the service was great"}
)

print(result)