import streamlit as st
from rag_utils import (
    ask_question,
    clear_all_data,
    clear_file_data,
    display_source,
    list_indexed_sources,
    load_conversation_history,
    process_files,
)

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
top_k = st.sidebar.number_input("Documents to Retrieve per Query", min_value=1, max_value=20, value=5)
answer_detail = st.sidebar.selectbox(
    "Answer Detail",
    ["Deep Dive", "Detailed", "Balanced"],
)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
max_tokens = st.sidebar.slider("Max Answer Tokens", min_value=512, max_value=4096, value=3072, step=256)

st.sidebar.divider()
st.sidebar.subheader("Indexed Data")
indexed_sources = list_indexed_sources()

if indexed_sources:
    selected_source = st.sidebar.selectbox(
        "Clear one uploaded file",
        indexed_sources,
        format_func=display_source,
    )

    if st.sidebar.button("Clear Selected File"):
        result = clear_file_data(selected_source)
        if result["success"]:
            st.sidebar.success(f"✅ {result['message']}")
        else:
            st.sidebar.error(f"❌ {result['message']}")
else:
    st.sidebar.caption("No indexed files found yet.")

confirm_clear_all = st.sidebar.checkbox("Confirm clear all indexed data")
if st.sidebar.button("Clear All Indexed Data", disabled=not confirm_clear_all):
    result = clear_all_data()
    if result["success"]:
        st.sidebar.success(f"✅ {result['message']}")
    else:
        st.sidebar.error(f"❌ {result['message']}")

st.sidebar.divider()

if st.sidebar.button("Submit & Process"):
    if uploaded_files:
        with st.spinner("🔄 Processing documents... Please wait"):
            result = process_files(uploaded_files, chunk_size, chunk_overlap)

        if result["success"]:
            st.success(f"✅ {result['message']}")
        else:
            st.error(f"❌ {result['message']}")
    else:
        st.warning("⚠️ Please upload at least one document.")

# Main Chat UI
st.subheader("💬 Ask a Question")
query = st.text_input("Enter your question here")

if st.button("Ask"):
    if query:
        with st.spinner("🤔 Finding the best answer..."):
            answer, sources = ask_question(
                query,
                k=top_k,
                temperature=temperature,
                answer_detail=answer_detail,
                max_tokens=max_tokens,
            )
        st.markdown("**Answer:**")
        st.markdown(answer)

        if sources:
            st.write("### Sources")
            for src in sources:
                st.json(src)
        else:
            st.info("No source chunks were returned for this answer.")
    else:
        st.warning("⚠️ Please enter a question.")

# Conversation history
st.write("### Conversation History")
history = load_conversation_history()
st.json(history) 
