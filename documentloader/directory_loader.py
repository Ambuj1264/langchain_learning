import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
book_path = os.path.join(script_dir, "book")

# Load PDF files
pdf_loader = DirectoryLoader(
    path=book_path,
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# Load text files
text_loader = DirectoryLoader(
    path=book_path,
    glob="*.txt",
    loader_cls=TextLoader
)

# Combine all documents
pdf_documents = pdf_loader.lazy_load()
text_documents = text_loader.lazy_load()
# all_documents = list(pdf_documents) + list(text_documents)

# print(f"Loaded {(pdf_documents)} PDF documents and {(text_documents)} text documents")
# print(f"Total documents: {len(pdf_documents) + len(text_documents)}")

for doc in pdf_documents:
    print(doc.metadata)
    print(doc.page_content)
    break

for doc in text_documents:
    print(doc.metadata)
    print(doc.page_content)
    break
