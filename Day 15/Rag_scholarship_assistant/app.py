import os
import json
import streamlit as st
import requests
from pypdf import PdfReader

st.set_page_config(page_title="AICTE Scholarship Assistant", layout="wide")

st.title("🎓 AICTE Scholarship Assistant (Simple RAG)")
st.write("Ask any questions related to Pragati and Saksham Scholarship Schemes.")

# Load .env file manually
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key] = val.strip()

api_key = os.environ.get("GROQ_API_KEY")

DATA_DIR = "data"

# Helper function to extract text from all PDFs in data folder
def load_and_index_documents():
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        st.error(f"No PDF files found in the '{DATA_DIR}' directory.")
        return []
    
    documents = []
    pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    
    progress_bar = st.progress(0)
    for i, pdf_name in enumerate(pdfs):
        pdf_path = os.path.join(DATA_DIR, pdf_name)
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    documents.append({
                        "text": text,
                        "source": pdf_name,
                        "page": page_num + 1
                    })
        except Exception as e:
            st.warning(f"Could not read {pdf_name}: {e}")
        progress_bar.progress((i + 1) / len(pdfs))
    progress_bar.empty()
    return documents

# Simple keyword-based retrieval (zero dependencies, extremely fast)
def retrieve_relevant_pages(query, documents, k=5):
    query_words = [word.lower() for word in query.split() if len(word) > 2]
    scored_docs = []
    
    for doc in documents:
        score = 0
        doc_text_lower = doc["text"].lower()
        for word in query_words:
            # Add score based on frequency of query words in the page
            score += doc_text_lower.count(word)
        scored_docs.append((score, doc))
        
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Return top K documents
    return [doc for score, doc in scored_docs[:k]]

# Initialize documents session state
if "documents" not in st.session_state:
    st.session_state["documents"] = []

st.sidebar.subheader("Document Processing")
if st.sidebar.button("Process Documents"):
    with st.spinner("Extracting text from PDFs..."):
        st.session_state["documents"] = load_and_index_documents()
        if st.session_state["documents"]:
            st.success(f"Successfully processed {len(st.session_state['documents'])} pages from PDFs!")

# Main QA Interface
if st.session_state["documents"]:
    st.markdown("### Ask a Question")
    user_query = st.text_input("Example: Who is eligible for the Pragati Scholarship?")
    
    if st.button("Get Answer"):
        if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE":
            st.error("Groq API Key not found. Please set `GROQ_API_KEY` in the `.env` file.")
        elif not user_query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching documents & generating answer with Groq (LLaMA-3.1)..."):
                try:
                    # 1. Retrieve the top relevant pages
                    relevant_pages = retrieve_relevant_pages(user_query, st.session_state["documents"], k=5)
                    
                    # 2. Build the context string
                    context = ""
                    for i, page in enumerate(relevant_pages):
                        context += f"\n--- Source {i+1}: {page['source']} (Page {page['page']}) ---\n"
                        context += page["text"] + "\n"
                    
                    # 3. Build the prompt
                    prompt = f"""
You are an AICTE Scholarship Assistant. Use the following retrieved context to answer the student's question accurately.
If you don't know the answer or the context doesn't contain the answer, say "I cannot find this information in the official documents."

Context:
{context}

Question:
{user_query}

Answer:
"""
                    # 4. Call Groq API using HTTP
                    url = "https://api.groq.com/openai/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ]
                    }
                    
                    response = requests.post(url, headers=headers, json=payload)
                    
                    if response.status_code == 200:
                        res_json = response.json()
                        try:
                            answer_text = res_json["choices"][0]["message"]["content"]
                            
                            st.markdown("### Answer")
                            st.info(answer_text)
                            
                            # Show Sources
                            with st.expander("View Retrieved Source Pages"):
                                for i, page in enumerate(relevant_pages):
                                    st.markdown(f"**Source {i+1}:** {page['source']} (Page {page['page']})")
                                    st.write(page["text"][:500] + "...")
                                    st.markdown("---")
                        except (KeyError, IndexError):
                            st.error("API returned an unexpected response format.")
                            st.json(res_json)
                    else:
                        st.error(f"Failed to generate content: {response.status_code} - {response.text}")
                            
                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.info("Please process the documents first using the button in the sidebar.")
