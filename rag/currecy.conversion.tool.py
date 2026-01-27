from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv
import requests

load_dotenv()

# --- Tool Definitions ---

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b 
@tool
def conversion_factor(from_currency: str, to_currency: str) -> float:
    """Get the currency conversion factor from one currency to another using an external API."""
    # Using the provided API Key
    api_key = "28d85836ba26998c91ca828a"
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_currency}/{to_currency}"
    
    response = requests.get(url)
    data = response.json()
    
    if response.status_code != 200:
        return f"Error fetching rate: {data.get('error-type', 'Unknown error')}"
        
    return data['conversion_rate']

# --- Main Logic ---

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Bind the tools we want to use (including the new conversion tool)
llm_with_tool = llm.bind_tools([multiply, conversion_factor])

# Ask a question that requires currency conversion
question = "Convert 100 USD to INR."
messages = [HumanMessage(content=question)]

response = llm_with_tool.invoke(messages)

print(f"User Question: {question}")
print("\n--- LLM decided to call ---")
print(response.tool_calls)

messages.append(response)

# Check if there are tool calls and execute them
# Loop to handle multiple tool calls (e.g. Rate -> Multiply)
while response.tool_calls:
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        tool_result = None

        if tool_name == "multiply":
            tool_result = multiply.invoke(tool_args)
        elif tool_name == "conversion_factor":
            rate = conversion_factor.invoke(tool_args)
            tool_result = f"The conversion rate from {tool_args['from_currency']} to {tool_args['to_currency']} is {rate}."
            
        print(f"--- Tool Result ({tool_name}): {tool_result} ---")
        
        if tool_result is not None:
            messages.append(ToolMessage(tool_call_id=tool_id, content=str(tool_result)))

    # Call the LLM again with the new history
    response = llm_with_tool.invoke(messages)
    messages.append(response)
    
    if response.tool_calls:
        print("\n--- LLM decided to call another tool ---")
        print(response.tool_calls)

print("\n--- Final Answer ---")
print(response.content)
