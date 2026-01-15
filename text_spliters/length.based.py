from langchain.text_splitter import CharacterTextSplitter

text = """
A text splitter is a tool or component that breaks large blocks of text into smaller, manageable "chunks" to overcome limitations of language models (LLMs) and improve readability/processing for AI applications like RAG (Retrieval-Augmented Generation), ensuring context is maintained and irrelevant data is excluded. It uses strategies like splitting by characters, words, sentences, or paragraphs, often with overlap, to create semantically coherent segments that fit within model context windows. 
Key Functions & Benefits
Fits Context Windows: LLMs have limits on how much text they can process at once; splitters break down documents to fit.
Improves Retrieval: Smaller, relevant chunks are easier to search and retrieve from vector databases, leading to more accurate results.
Maintains Context: Advanced splitters use overlap (e.g., 30 characters) and hierarchical splitting (paragraphs > sentences > words) to keep meaning intact.
Enhances Readability: For web content, splitting makes text digestible and improves user experience. 
Common Strategies
Character Text Splitter: Splits by a specific character (e.g., space, newline).
RecursiveCharacterTextSplitter: Tries larger separators first (like \n\n), then smaller ones (like \n, then space), ensuring semantic breaks are respected.
Token-Based Splitters: Splits based on the model's actual tokens, ensuring perfect alignment with the model's understanding. 
Example Use Case (LangChain)
In a RAG system, a Text Splitter takes a long PDF, breaks it into chunks (e.g., 400 characters with 30 characters overlap), converts these chunks to vectors (embeddings), stores them in a vector database, and retrieves relevant chunks to answer user queries. 
Text splitters - Docs by LangChain
Copy page. pip. uv. pip install -U langchain-text-splitters. Text splitters break large docs into smaller chunks that will be retr...

LangChain

Text Splitter in LangChain - GeeksforGeeks
4 Nov 2025 — Working with large documents or unstructured text often creates challenges for language models, as they can only process limited t...

GeeksforGeeks

Learn how to use text splitters in LangChain - Web3Dave
12 Mar 2024 — Now here is what's happening: Load and Parse the PDF. Set Up Text Splitter: Create an instance of the RecursiveCharacterTextSplitt...

blog.davideai.dev

Show all

"""



splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)


result = splitter.split_text(text)

print(result)
