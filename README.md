# RAG Supported Study AI

A beginner-friendly AI study assistant that lets users upload study files and ask questions about them.

This project uses **RAG**: Retrieval-Augmented Generation.  
That means the app first searches your uploaded files for useful information, then uses an AI model to answer based on that file content.

---

## What This Project Does

Users can:

- Upload study documents
- Ask questions about uploaded files
- Get AI answers based on the document content
- Use it from a simple Streamlit web interface
- Study faster by chatting with PDFs, text files, CSV files, and other supported documents

Example questions:

```text
Summarize this chapter.
What are the main points in this PDF?
Explain this topic like I am a beginner.
Make 10 exam questions from this file.
Find important definitions from the uploaded document.
```

---

## Tech Stack

This project uses:

- **Python** for backend logic
- **Streamlit** for the web app
- **LangChain** for RAG pipeline
- **Google Gemini API** for AI answers
- **FAISS / vector database** for document search
- **PDF / CSV / TXT loaders** for reading files

---

## Project Folder Structure

Your project may look something like this:

```text
ragsupportedstudyai/
│
├── ui.py
├── rag_utils.py
├── requirements.txt
├── .env
├── README.md
└── vector_store/
```

Main files:

| File | Purpose |
|---|---|
| `ui.py` | Streamlit frontend/interface |
| `rag_utils.py` | RAG logic, file processing, AI answering |
| `requirements.txt` | Python packages needed |
| `.env` | Stores your API key safely |
| `README.md` | Project explanation and usage guide |

---

# Beginner Step-by-Step Setup

Follow these steps carefully.

---

## 1. Install Python

Make sure Python is installed.

Check by running:

```bash
python --version
```

or:

```bash
python3 --version
```

You should see something like:

```text
Python 3.10+
```

Recommended: Python **3.10, 3.11, or 3.12**.

---

## 2. Download or Clone the Project

If the project is on GitHub, clone it:

```bash
git clone https://github.com/YOUR_USERNAME/ragsupportedstudyai.git
cd ragsupportedstudyai
```

If you already have the folder, just open the project folder in your terminal.

---

## 3. Create a Virtual Environment

A virtual environment keeps your project packages separate from your system packages.

### On Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### On Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

When it works, your terminal will show something like:

```text
(.venv)
```

---

## 4. Install Required Packages

Run:

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` yet, create one and add:

```text
streamlit
python-dotenv
pandas
pypdf
faiss-cpu
langchain
langchain-community
langchain-google-genai
google-generativeai
```

Then install again:

```bash
pip install -r requirements.txt
```

---

## 5. Add Your Gemini API Key

You need an API key so the app can use the AI model.

Create a file named:

```text
.env
```

Inside `.env`, add:

```env
GOOGLE_API_KEY=your_api_key_here
```

Example:

```env
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXX
```

Important:

- Do not share your API key publicly.
- Do not upload `.env` to GitHub.
- Add `.env` inside `.gitignore`.

Your `.gitignore` should include:

```text
.env
.venv/
__pycache__/
vector_store/
```

---

## 6. Run the Streamlit App

Run this command from the project folder:

```bash
streamlit run ui.py
```

After running, Streamlit will show a local link like:

```text
http://localhost:8501
```

Open that link in your browser.

---

# How to Use the App

## Step 1: Open the App

Run:

```bash
streamlit run ui.py
```

Then open the browser link.

---

## Step 2: Upload Your Files

In the sidebar, upload your study files.

Supported file types may include:

```text
PDF
TXT
CSV
DOCX
```

Upload one or more files.

---

## Step 3: Process the Files

Click:

```text
Submit & Process
```

The app will:

1. Read your files
2. Split them into smaller chunks
3. Convert chunks into embeddings
4. Store them in a searchable vector database

This is what allows the AI to find answers from your files.

---

## Step 4: Ask Questions

After processing, type a question like:

```text
Explain this document in simple words.
```

or:

```text
Give me important exam questions from this PDF.
```

The app will search your uploaded files and generate an answer.

---

## Step 5: Study With It

You can ask:

```text
Summarize this topic.
Make notes from this file.
Explain like I am 10 years old.
Create MCQs.
Create flashcards.
Find definitions.
Make a revision plan.
```

---

# Example Use Cases

## For Students

```text
Upload textbook PDF → ask for chapter summary
Upload notes → ask for exam questions
Upload CSV data → ask for table summary
Upload assignment file → ask for explanation
```

## For Teachers

```text
Upload study material → create quiz questions
Upload syllabus → make study plan
Upload topic notes → generate explanations
```

## For Self-Learning

```text
Upload any document → learn by asking questions
Upload research papers → simplify complex ideas
Upload reports → extract key points
```

---

# Common Problems and Fixes

## Problem: `ModuleNotFoundError`

Example:

```text
ModuleNotFoundError: No module named 'langchain_google_genai'
```

Fix:

```bash
pip install langchain-google-genai
```

If many packages are missing, run:

```bash
pip install -r requirements.txt
```

---

## Problem: `No module named langchain.chains`

This usually happens because LangChain versions changed.

Try updating packages:

```bash
pip install -U langchain langchain-community langchain-google-genai
```

If your code uses old imports, update them according to the current package structure.

---

## Problem: API Key Error

Check that your `.env` file exists and contains:

```env
GOOGLE_API_KEY=your_api_key_here
```

Also make sure your code loads it with:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Problem: Streamlit Shows Blank Page

Try these checks:

1. Look at the terminal for errors.
2. Make sure `ui.py` is saved.
3. Make sure your virtual environment is active.
4. Run:

```bash
streamlit run ui.py
```

5. If imports are failing, install missing packages.

---

## Problem: File Upload Works But Answers Are Bad

Possible reasons:

- File was not processed correctly
- Chunk size is too small or too large
- API key/model is not working properly
- The question is too vague
- The uploaded document does not contain the answer

Try asking a more specific question:

```text
According to the uploaded PDF, what is the definition of inflation?
```

Instead of:

```text
Tell me everything.
```

---

# Suggested Questions to Try

After uploading a file, try these:

```text
Summarize this document in 10 bullet points.
Explain the hardest topic in simple language.
Make 15 MCQs with answers.
Make short notes for exam revision.
What are the most important definitions?
Create flashcards from this document.
Give me a 7-day study plan from this file.
Find all formulas in this document.
Explain this like I am a beginner.
```

---

# How to Push This Project to GitHub

Run these commands from the project folder.

## 1. Initialize Git

```bash
git init
```

## 2. Add Files

```bash
git add .
```

## 3. Commit

```bash
git commit -m "Add RAG supported study AI project"
```

## 4. Connect GitHub Repository

Create a new GitHub repository named:

```text
ragsupportedstudyai
```

Then connect it:

```bash
git remote add origin https://github.com/YOUR_USERNAME/ragsupportedstudyai.git
```

## 5. Push

```bash
git branch -M main
git push -u origin main
```

---

# Deployment Options

You can deploy this project using free or beginner-friendly platforms such as:

- Streamlit Community Cloud
- Hugging Face Spaces
- Render free tier, if available
- Local hosting on your own PC

For the easiest beginner deployment, use **Streamlit Community Cloud**.

Basic deployment steps:

1. Push project to GitHub
2. Open Streamlit Community Cloud
3. Connect your GitHub account
4. Select this repository
5. Set the main file as:

```text
ui.py
```

6. Add your secret API key in the platform secrets/settings
7. Deploy the app

---

# Important Security Notes

Never upload these to GitHub:

```text
.env
.venv/
API keys
Private files
Personal documents
```

Your `.gitignore` should protect them.

---

# Future Improvements

You can improve this project by adding:

- Login system
- User history
- Chat memory
- Better UI design
- PDF page citations
- File source preview
- Multiple AI model support
- Free usage limits
- Ads for monetization
- Admin dashboard
- Public deployment

---

# Final Summary

**RAG Supported Study AI** is a study assistant that lets users upload documents and chat with them.

It is useful for:

- Students
- Teachers
- Researchers
- Self-learners
- Exam preparation
- Quick document understanding

The goal is simple:

```text
Upload files → Ask questions → Get useful study answers
```
