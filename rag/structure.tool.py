from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(description="The first number to multiply")
    b: int = Field(description="The second number to multiply")

def multiply_function(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply_function,
    name="multiply",
    description="Use this tool to multiply two numbers",
    args_schema=MultiplyInput
)

print(multiply_tool.invoke({"a": 2, "b": 3}))
