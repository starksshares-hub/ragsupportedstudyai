import streamlit as st
from rag_utils import process_files, ask_question, load_conversation_history

# Set Even Loop  
import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(page_title="Advance Rag", layout="wide")
st.title("📄 Advance Rag | Chat with Your Files ")

# Sidebar for file upload and settings
st.sidebar.header("Configuration")
uploaded_files = st.sidebar.file_uploader(
    "Upload your documents (PDF, CSV, TXT)", 
    type=["pdf", "csv", "txt"],
    accept_multiple_files=True
)
chunk_size = st.sidebar.number_input("Chunk Size", min_value=200, max_value=2000, value=1000, step=100)
chunk_overlap = st.sidebar.number_input("Chunk Overlap", min_value=0, max_value=500, value=100, step=10)
top_k = st.sidebar.number_input("Documents to Retrieve per Query", min_value=1, max_value=10, value=3)

if st.sidebar.button("Submit & Process"):
    if uploaded_files:
        with st.spinner("🔄 Processing documents... Please wait"):
            process_files(uploaded_files, chunk_size, chunk_overlap)
        st.success("✅ Documents processed and stored in local vector DB")
    else:
        st.warning("⚠️ Please upload at least one document.")

# Main Chat UI
st.subheader("💬 Ask a Question")
query = st.text_input("Enter your question here")

if st.button("Ask"):
    if query:
        with st.spinner("🤔 Finding the best answer..."):
            answer, sources = ask_question(query, top_k)
        st.markdown(f"**Answer:** {answer}")

        st.write("### Sources")
        for src in sources:
            st.json(src)
    else:
        st.warning("⚠️ Please enter a question.")

# Conversation history
st.write("### Conversation History")
history = load_conversation_history()
st.json(history) 