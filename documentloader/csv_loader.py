import os
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_dir, "./book/customers-100.csv")

# Load CSV file - each row becomes a separate document
loader = CSVLoader(file_path=csv_file_path)
documents = loader.load()

# Display loaded documents
print("=" * 50)
print("CSV DOCUMENTS LOADED")
print("=" * 50)
print(f"Total documents (rows): {len(documents)}\n")

for i, doc in enumerate(documents):
    print(f"--- Document {i + 1} ---")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create a prompt template for analyzing CSV data
prompt_template = PromptTemplate(
    input_variables=["data", "question"],
    template="""You are a helpful assistant that analyzes CSV data and answers questions.

CSV Data:
{data}

Question: {question}

Provide a clear and concise answer based on the CSV data above.
"""
)

# Create the chain
parser = StrOutputParser()
chain = prompt_template | llm | parser

# Combine all document contents for context
all_data = "\n\n".join([doc.page_content for doc in documents])

# Interactive Q&A loop
print("=" * 50)
print("CSV Data loaded! Ask questions about the data (type 'exit' to quit):")
print("=" * 50)

while True:
    question = input("\nYour question: ")
    if question.lower() == 'exit':
        print("Goodbye!")
        break
    
    response = chain.invoke({
        "data": all_data,
        "question": question
    })
    print(f"\nAnswer: {response}")
