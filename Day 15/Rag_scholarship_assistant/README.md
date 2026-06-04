# AICTE Scholarship Assistant (RAG)

A simple Retrieval-Augmented Generation (RAG) based Scholarship Assistant for AICTE Pragati and Saksham scholarship schemes.

## Features
- Processes PDF documents from the `data/` folder.
- Uses `sentence-transformers` (all-MiniLM-L6-v2) to generate embeddings locally (free & fast).
- Stores embeddings in a local Chroma vector database.
- Uses Google Gemini API (`gemini-1.5-flash`) for generating accurate answers based on the retrieved context.
- Simple, user-friendly Streamlit web interface.

## Setup Instructions

1. **Install Dependencies**
   Make sure you have Python installed. Run the following command to install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   Start the Streamlit app using:
   ```bash
   streamlit run app.py
   ```

3. **Using the Assistant**
   - Open the provided local URL in your browser (usually http://localhost:8501).
   - Get a free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
   - Enter your API Key in the sidebar.
   - Click **"Process Documents"** in the sidebar. Wait for the vector database to be created.
   - Ask any question in the main interface!

## Example Questions
- "Who is eligible for the Pragati Scholarship?"
- "What is the family income limit for Saksham Scholarship?"
- "What are the required documents for renewal?"
