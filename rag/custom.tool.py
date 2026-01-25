from langchain_core.tools import tool



# step 1 create a function

def multiply(x, y):
    return x * y

#  step 2 add type hints
def multiply(x: int, y: int) -> int:
    return x * y

# step 3 add tool decorator

@tool
def multiply(a:int , b:int) -> int:
    """Multiply two numbers together."""
    return a * b


result = multiply.invoke({"a": 2, "b": 3})
print(result, multiply.description, multiply.name, multiply.args)
