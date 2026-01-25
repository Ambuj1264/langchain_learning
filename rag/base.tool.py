from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type


class MultiplyInput(BaseModel):
    a: int = Field(description="The first number to multiply")
    b: int = Field(description="The second number to multiply")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Use this tool to multiply two numbers"
    

    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b


result = MultiplyTool().invoke({"a": 2, "b": 3})
print(result)




