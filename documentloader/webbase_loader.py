import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Set USER_AGENT to avoid warning
os.environ["USER_AGENT"] = "Mozilla/5.0"

url = "https://www.flipkart.com/kaspy-printed-typography-men-round-neck-grey-t-shirt/p/itm901116e619f3e?pid=TSHGS588HEHSDJ6X&lid=LSTTSHGS588HEHSDJ6X7OOHTQ&marketplace=FLIPKART&store=clo%2Fash%2Fank%2Fedy&srno=b_1_4&otracker=browse&fm=organic&iid=b0545e69-e7ca-4c38-81f7-fa11fe262c92.TSHGS588HEHSDJ6X.SEARCH&ppt=browse&ppn=browse&ssid=csu0rnyf400000001768127551048"

# Load the web page content
loader = WebBaseLoader(url)
documents = loader.load()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create a prompt template for Q&A
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that answers questions based on the provided product information.

Product Information:
{context}

Question: {question}

Answer the question based only on the product information provided above. If the information is not available, say so.
"""
)

# Create the chain
chain = prompt_template | llm

# Interactive Q&A loop
print("Product loaded successfully! Ask questions about the product (type 'exit' to quit):\n")

while True:
    question = input("Your question: ")
    if question.lower() == 'exit':
        print("Goodbye!")
        break
    
    response = chain.invoke({
        "context": documents[0].page_content,
        "question": question
    })
    print(f"\nAnswer: {response.content}\n")
