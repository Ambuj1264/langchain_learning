import os
from langchain_community.document_loaders import PyPDFLoader

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "sample.pdf")

loader = PyPDFLoader(file_path)
documents = loader.load()
print(documents[0].page_content, "---------------\n", documents[0].metadata)
