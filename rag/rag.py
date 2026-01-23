# ===================== IMPORTS =====================
from dotenv import load_dotenv
load_dotenv()

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


# ===================== CONFIG =====================
VIDEO_ID = "l71aOtTJ1gE"

# ===================== GET TRANSCRIPT =====================
try:
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(VIDEO_ID, languages=["en", "hi"])

    transcript_text = " ".join(snippet.text for snippet in transcript.snippets)
    print("Transcript fetched successfully\n", transcript_text)

except TranscriptsDisabled:
    raise Exception("❌ No captions available for this video")

# ===================== TEXT SPLITTING =====================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

documents = splitter.split_documents([
    Document(page_content=transcript_text)
])

# ===================== EMBEDDINGS + VECTOR STORE =====================
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
# print("===============", vectorstore, "========vector store")

# ===================== PROMPT =====================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""
)

# ===================== LLM =====================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ===================== QUERY FUNCTION =====================
def ask_question(question: str):
    docs = vectorstore.similarity_search(question, k=4)
    print("===============", docs[0].page_content, "========docs")
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content

# ===================== TEST QUERY =====================
if __name__ == "__main__":
    answer = ask_question("What is this video mainly about?")
    print("\nAnswer:\n", answer)
