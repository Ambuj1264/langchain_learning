from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv

load_dotenv()




# tool create

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b+80


# print(multiply.invoke({"a": 2, "b": 3}))


# tool binding

llm = ChatOpenAI(model= "gpt-4o-mini", temperature=0)

llm_with_tool = llm.bind_tools([multiply])




messages = [HumanMessage(content="Calculate the square of 5 using the tool and return the exact result given by the tool, even if it seems incorrect.")]
response = llm_with_tool.invoke(messages)

print("--- LLM decided to call ---")
print(response.tool_calls)

messages.append(response)

# Check if there are tool calls
if response.tool_calls:
    for tool_call in response.tool_calls:
        # 1. Execute the tool
        if tool_call["name"] == "multiply":
            tool_result = multiply.invoke(tool_call["args"])
            print(f"--- Tool Result: {tool_result} ---")
            
            # 2. Create a ToolMessage with the result
            messages.append(ToolMessage(tool_call_id=tool_call["id"], content=str(tool_result)))

    # 3. Call the LLM again with the tool result to get the final answer
    final_response = llm_with_tool.invoke(messages)
    print("\n--- Final Answer ---")
    print(final_response.content)



