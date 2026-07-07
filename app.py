import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
from pypdf import PdfReader
from docx import Document
import urllib.request
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DataIntern - RAG CRM Assistant", layout="wide")
st.title("💼 DataIntern: RAG Chatbot for CRM & Business Data")
st.caption("Secure Multi-Format Ingestion & Instant Visualization Engine")

# --- SECURE SECRETS LOADING ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GOOGLE_DRIVE_API_KEY = st.secrets["GOOGLE_DRIVE_API_KEY"]
    DRIVE_FOLDER_ID = st.secrets["DRIVE_FOLDER_ID"]
except KeyError as e:
    st.error(f"Missing Secret: {e}. Please add it to your Streamlit Cloud Secrets settings.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- INITIALIZE SESSION STATE ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration Panel")
    st.success("✅ API Keys securely loaded from backend.")
    fetch_btn = st.button("Ingest Files From Drive")

    st.markdown("---")
    st.markdown("### 📋 System Status")
    if st.session_state.processed_files:
        st.success(f"Ingested {len(st.session_state.processed_files)} files.")
        for f in st.session_state.processed_files:
            st.text(f"• {f}")
    else:
        st.info("No documents currently loaded into Vector Store.")

# --- HELPERS: PURE PYTHON VECTOR MATH ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- GOOGLE DRIVE FILE INGESTION ---
def fetch_files_from_drive(folder_id, api_key):
    url = f"https://www.googleapis.com/drive/v3/files?q='{folder_id}'+in+parents&key={api_key}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data.get('files', [])
    except Exception as e:
        st.sidebar.error(f"Failed to access Google Drive. Check Drive API Key and Folder Permissions.\nError: {e}")
        return []

def download_drive_file(file_id, api_key):
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            return response.read()
    except Exception:
        return None

# --- PARSING ENGINE FOR MULTI-FORMATS ---
def parse_file_content(file_name, file_bytes):
    chunks = []
    ext = file_name.split('.')[-1].lower()
    try:
        if ext == 'csv':
            df = pd.read_csv(io.BytesIO(file_bytes))
            for idx, row in df.iterrows():
                row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
                chunks.append({"text": f"Row {idx}: {row_str}", "source": f"{file_name} (Row {idx})"})
        elif ext in ['xlsx', 'xls']:
            xl = pd.ExcelFile(io.BytesIO(file_bytes))
            for sheet in xl.sheet_names:
                df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
                for idx, row in df.iterrows():
                    row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
                    chunks.append({"text": f"Sheet: {sheet} | Row {idx}: {row_str}", "source": f"{file_name} ({sheet}, R{idx})"})
        elif ext == 'pdf':
            pdf_reader = PdfReader(io.BytesIO(file_bytes))
            for idx, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    chunks.append({"text": text, "source": f"{file_name} (Page {idx+1})"})
        elif ext in ['docx', 'doc']:
            doc = Document(io.BytesIO(file_bytes))
            for idx, para in enumerate([p.text for p in doc.paragraphs if p.text.strip()]):
                chunks.append({"text": para, "source": f"{file_name} (Para {idx+1})"})
        elif ext == 'json':
            data = json.loads(file_bytes.decode('utf-8'))
            chunks.append({"text": json.dumps(data, indent=2), "source": file_name})
    except Exception as e:
        st.warning(f"Could not parse {file_name}: {e}")
    return chunks

# --- EXECUTE INGESTION PIPELINE ---
if fetch_btn:
    with st.spinner("Accessing Google Drive & compiling documents..."):
        files = fetch_files_from_drive(DRIVE_FOLDER_ID, GOOGLE_DRIVE_API_KEY)
        if not files:
            st.sidebar.warning("No files found or folder is not public.")
        else:
            all_chunks = []
            processed_names = []
            
            for f in files:
                f_name, f_id = f['name'], f['id']
                f_bytes = download_drive_file(f_id, GOOGLE_DRIVE_API_KEY)
                if f_bytes:
                    file_chunks = parse_file_content(f_name, f_bytes)
                    all_chunks.extend(file_chunks)
                    processed_names.append(f_name)
            
            if all_chunks:
                with st.spinner("Generating embeddings..."):
                    texts = [c['text'] for c in all_chunks]
                    # FIX: Switched to universally supported legacy embedding model
                    response = genai.embed_content(
                        model="models/embedding-001",
                        content=texts,
                        task_type="retrieval_document"
                    )
                    
                    st.session_state.vectorstore = []
                    for i, embedding in enumerate(response['embedding']):
                        st.session_state.vectorstore.append({
                            "vector": embedding,
                            "text": all_chunks[i]['text'],
                            "source": all_chunks[i]['source']
                        })
                st.session_state.processed_files = processed_names
                st.rerun()

# --- CHAT INTERFACE & ENGINE ---
# Render historical turns
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("html_chart"):
            components.html(message["html_chart"], height=500)

if user_query := st.chat_input("Ask DataIntern about your business logs..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    if st.session_state.vectorstore:
        context_str = ""
        # FIX: Switched to universally supported legacy embedding model
        query_embedding = genai.embed_content(
            model="models/embedding-001",
            content=user_query,
            task_type="retrieval_query"
        )['embedding']
        
        # Rank entries
        scored = [(cosine_similarity(query_embedding, item["vector"]), item) for item in st.session_state.vectorstore]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = scored[:15]
        
        context_blocks = [f"Source [{item['source']}]: {item['text']}\n" for score, item in top_k]
        context_str = "\n".join(context_blocks)
        
        system_prompt = f"""
        You are DataIntern, a RAG engine.
        1. Answer based ONLY on the context. If not found, say "I don't see that in your files."
        2. Format output strictly as JSON.

        If text answer:
        {{
            "requires_chart": false,
            "text_response": "Your answer here",
            "citations": ["Source 1", "Source 2"]
        }}

        If user asks for a chart:
        {{
            "requires_chart": true,
            "text_response": "Here is the chart:",
            "citations": ["Source"],
            "chart_data": {{
                "type": "bar", // or line, pie, scatter
                "title": "Chart Title",
                "x_label": "X Axis",
                "y_label": "Y Axis",
                "x_data": ["A", "B"],
                "y_data": [10, 20]
            }}
        }}

        CONTEXT:
        {context_str}
        """
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content([system_prompt, user_query])
                    
                    raw_response = response.text.strip()
                    if raw_response.startswith("```json"):
                        raw_response = raw_response[7:-3]
                    elif raw_response.startswith("```"):
                        raw_response = raw_response[3:-3]
                    
                    response_data = json.loads(raw_response)
                    text = response_data.get("text_response", "Error parsing text.")
                    citations = response_data.get("citations", [])
                    
                    final_text = f"{text}\n\n*Sources: {', '.join(citations)}*" if citations else text
                    st.markdown(final_text)
                    
                    html_str = None
                    if response_data.get("requires_chart") and response_data.get("chart_data"):
                        c = response_data["chart_data"]
                        fig = go.Figure()
                        ctype, x, y = c.get("type", "bar"), c.get("x_data", []), c.get("y_data", [])
                        
                        if ctype == "bar": fig.add_trace(go.Bar(x=x, y=y))
                        elif ctype == "line": fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers'))
                        elif ctype == "pie": fig.add_trace(go.Pie(labels=x, values=y))
                        elif ctype == "scatter": fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))
                        
                        fig.update_layout(title=c.get("title", ""), xaxis_title=c.get("x_label", ""), yaxis_title=c.get("y_label", ""))
                        html_str = fig.to_html(full_html=True, include_plotlyjs='cdn')
                        components.html(html_str, height=500)
                    
                    st.session_state.messages.append({"role": "assistant", "content": final_text, "html_chart": html_str})

                except Exception as e:
                    st.error(f"Failed to parse LLM response. Ensure context contains enough data. {e}")
    elif not st.session_state.vectorstore:
        st.info("Please ingest a Google Drive folder in the sidebar to start.")
