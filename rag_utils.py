import os
import json
import tempfile
from functools import lru_cache

import chromadb
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")
HISTORY_FILE = os.path.join(VECTOR_STORE_PATH, "conversation_history.json")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "langchain"
ANSWER_DETAIL_GUIDES = {
    "Balanced": (
        "Write a clear answer with 3 to 5 focused paragraphs. "
        "Include the most important supporting details from the sources."
    ),
    "Detailed": (
        "Write a fuller answer with clear sections. Aim for 6 to 9 substantial "
        "paragraphs or bullets when the context supports it. Include definitions, "
        "conditions, steps, exceptions, dates, numbers, and named items found in the sources."
    ),
    "Deep Dive": (
        "Write a comprehensive answer. Start with a short summary, then expand with "
        "well-labeled sections, detailed bullets, caveats, and source-backed examples. "
        "Aim for a long answer when the context supports it."
    ),
}

os.makedirs(VECTOR_STORE_PATH, exist_ok=True)


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"local_files_only": True}
        )
    except Exception:
        pass

    try:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the Hugging Face embedding model. "
            "Check your internet connection for the first download, then try again."
        ) from exc


def process_files(files, chunk_size=1000, chunk_overlap=100):
    docs = []
    errors = []

    for file in files:
        file_ext = os.path.splitext(file.name)[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            if hasattr(file, "seek"):
                file.seek(0)
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
                errors.append(f"{file.name}: unsupported file type.")
                continue

            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = file.name
            docs.extend(loaded_docs)

        except Exception as exc:
            errors.append(f"{file.name}: {exc}")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not docs:
        detail = " ".join(errors) if errors else "No readable content was found."
        return {
            "success": False,
            "message": f"No documents were processed. {detail}"
        }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(docs)

    try:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory=VECTOR_STORE_PATH,
            collection_name=CHROMA_COLLECTION_NAME
        )
    except Exception as exc:
        return {
            "success": False,
            "message": f"Documents were read, but indexing failed. Details: {exc}"
        }

    # For older Chroma versions
    if hasattr(vectordb, "persist"):
        vectordb.persist()

    message = f"Processed {len(docs)} documents into {len(chunks)} chunks."
    if errors:
        message += " Some files were skipped: " + " ".join(errors)

    return {"success": True, "message": message}


def ask_question(query, k=5, temperature=0.45, answer_detail="Deep Dive", max_tokens=3072):
    query = (query or "").strip()
    if not query:
        return "Please enter a question first.", []

    temperature = normalize_float(temperature, default=0.45, minimum=0.0, maximum=1.0)
    max_tokens = normalize_int(max_tokens, default=3072, minimum=512, maximum=4096)
    answer_detail = answer_detail if answer_detail in ANSWER_DETAIL_GUIDES else "Deep Dive"

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        return "GOOGLE_API_KEY not found. Add it inside your .env file.", []

    try:
        vectordb = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=get_embeddings(),
            collection_name=CHROMA_COLLECTION_NAME
        )

        if hasattr(vectordb, "_collection") and vectordb._collection.count() == 0:
            return "No documents are indexed yet. Upload files and click Submit & Process first.", []

        retriever = vectordb.as_retriever(
            search_kwargs={"k": max(1, int(k))}
        )
        source_documents = retriever.invoke(query)

    except Exception as exc:
        return f"I could not search the uploaded documents yet. Details: {exc}", []

    if not source_documents:
        return "I could not find any relevant text in the uploaded documents for that question.", []

    context = "\n\n".join(
        [
            (
                f"Source {i + 1} "
                f"({display_source(doc.metadata.get('source', 'uploaded document'))}):\n"
                f"{doc.page_content}"
            )
            for i, doc in enumerate(source_documents)
        ]
    )

    prompt = f"""
You are StudyTheFile, a warm and careful RAG assistant.

Use the uploaded document context to answer the user.

Detail level:
{ANSWER_DETAIL_GUIDES[answer_detail]}

Style:
- Be friendly, clear, and genuinely helpful.
- Give a direct answer first, then add the useful details.
- Prefer a richer answer over a short one when the sources contain enough material.
- Use headings and bullets when they make the answer easier to read.
- Pull in specific details from multiple source chunks instead of only summarizing one chunk.
- If the context only partially answers the question, say what is known and what is missing.
- If the answer is not in the context, say: "I could not find this in the uploaded documents."
- Do not invent facts that are not supported by the context.
- Do not pad the answer if the uploaded context is thin.

Context:
{context}

Question:
{query}

Answer:
"""

    sources = format_sources(source_documents)

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        response = llm.invoke(prompt)
    except Exception as exc:
        return f"I found relevant document text, but the Gemini answer call failed. Details: {exc}", sources

    answer = response.content if hasattr(response, "content") else str(response)
    answer = (answer or "").strip()

    if not answer:
        answer = "I found relevant document text, but Gemini returned an empty answer. Please try asking again."

    log_result(query, answer, source_documents)

    return answer, sources


def format_sources(source_documents):
    sources = [
        {
            "metadata": doc.metadata,
            "preview": doc.page_content[:300]
        }
        for doc in source_documents
    ]
    return sources


def normalize_float(value, default, minimum, maximum):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return min(max(value, minimum), maximum)


def normalize_int(value, default, minimum, maximum):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return min(max(value, minimum), maximum)


def list_indexed_sources():
    try:
        collection = get_chroma_collection()
        if collection is None or collection.count() == 0:
            return []

        result = collection.get(include=["metadatas"])
        sources = {
            metadata.get("source")
            for metadata in result.get("metadatas", [])
            if metadata and metadata.get("source")
        }
        return sorted(sources, key=lambda source: os.path.basename(str(source)).lower())
    except Exception:
        return []


def clear_file_data(source_name):
    if not source_name:
        return {"success": False, "message": "Choose a file to clear first."}

    try:
        collection = get_chroma_collection()
        if collection is None or collection.count() == 0:
            return {"success": False, "message": "No indexed documents were found."}

        result = collection.get(where={"source": source_name})
        ids = result.get("ids", [])
        if not ids:
            return {
                "success": False,
                "message": f"No indexed chunks were found for {display_source(source_name)}."
            }

        collection.delete(ids=ids)
        remove_history_for_source(source_name)

        return {
            "success": True,
            "message": f"Cleared {len(ids)} chunks for {display_source(source_name)}."
        }
    except Exception as exc:
        return {"success": False, "message": f"Could not clear that file. Details: {exc}"}


def clear_all_data():
    try:
        client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
        for collection in client.list_collections():
            client.delete_collection(collection.name)

        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)

        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        return {
            "success": True,
            "message": "Cleared all indexed documents and conversation history."
        }
    except Exception as exc:
        return {"success": False, "message": f"Could not clear indexed data. Details: {exc}"}


def get_chroma_collection():
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    try:
        return client.get_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        return None


def display_source(source_name):
    return os.path.basename(str(source_name)) or str(source_name)


def remove_history_for_source(source_name):
    if not os.path.exists(HISTORY_FILE):
        return

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    filtered_history = [
        entry
        for entry in history
        if not entry_uses_source(entry, source_name)
    ]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered_history, f, indent=2)


def entry_uses_source(entry, source_name):
    for source in entry.get("sources", []):
        metadata = source.get("metadata", {})
        if metadata.get("source") == source_name:
            return True
    return False


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
