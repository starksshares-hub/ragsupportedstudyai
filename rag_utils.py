import os
import json
import tempfile

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

VECTOR_STORE_PATH = "vector_store"
HISTORY_FILE = os.path.join(VECTOR_STORE_PATH, "conversation_history.json")

os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def process_files(files, chunk_size=1000, chunk_overlap=100):
    docs = []

    for file in files:
        file_ext = os.path.splitext(file.name)[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            if file_ext == ".pdf":
                loader = PyPDFLoader(tmp_path)

            elif file_ext == ".csv":
                loader = CSVLoader(tmp_path)

            elif file_ext == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")

            else:
                continue

            docs.extend(loader.load())

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not docs:
        return {
            "success": False,
            "message": "No supported documents found."
        }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )

    # For older Chroma versions
    if hasattr(vectordb, "persist"):
        vectordb.persist()

    return {
        "success": True,
        "message": f"Processed {len(docs)} documents into {len(chunks)} chunks."
    }


def ask_question(query, k=3):
    if not os.getenv("GOOGLE_API_KEY"):
        return "GOOGLE_API_KEY not found. Add it inside your .env file.", []

    vectordb = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": k}
    )

    source_documents = retriever.invoke(query)

    context = "\n\n".join(
        [
            f"Source {i + 1}:\n{doc.page_content}"
            for i, doc in enumerate(source_documents)
        ]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using only the context below.
If the answer is not present in the context, say:
"I could not find this in the uploaded documents."

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content if hasattr(response, "content") else str(response)

    log_result(query, answer, source_documents)

    sources = [
        {
            "metadata": doc.metadata,
            "preview": doc.page_content[:300]
        }
        for doc in source_documents
    ]

    return answer, sources


def log_result(query, answer, sources):
    entry = {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "metadata": doc.metadata,
                "preview": doc.page_content[:300]
            }
            for doc in sources
        ]
    }

    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []
    else:
        history = []

    history.append(entry)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def load_conversation_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    return []