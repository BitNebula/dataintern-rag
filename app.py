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
import urllib.error
import io
import time
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DataIntern - RAG CRM Assistant", layout="wide")
st.title("💼 DataIntern: RAG Chatbot for CRM & Business Data")
st.caption("Google Drive Multi-Format Ingestion & Visualization Engine")

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
    st.success("✅ Secure AI Core Initialized.")
    fetch_btn = st.button("Ingest Files From Drive", type="primary")

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
    # Prevent division by zero
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def chunk_text(text, source_name, max_words=500):
    chunks = []
    words = text.split()
    for i in range(0, len(words), max_words):
        chunk_str = " ".join(words[i:i+max_words])
        chunks.append({"text": chunk_str, "source": source_name})
    return chunks

# --- GOOGLE DRIVE FILE INGESTION ---
def fetch_files_from_drive(folder_id, api_key):
    url = f"https://www.googleapis.com/drive/v3/files?q='{folder_id}'+in+parents&key={api_key}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())
            return data.get('files', [])
    except Exception as e:
        return []

def download_drive_file(file_id, api_key):
    # BYPASS API RESTRICTIONS: Try direct web download link first for public files
    url_web = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        req = urllib.request.Request(url_web, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.read(), None
    except Exception:
        # Fallback to standard API if the web link is blocked
        url_api = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
        try:
            req = urllib.request.Request(url_api, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=20) as response:
                return response.read(), None
        except urllib.error.HTTPError as e:
            return None, f"HTTP {e.code}: Could not fetch bytes. File might not be fully public."
        except Exception as e:
            return None, str(e)

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
                    chunks.extend(chunk_text(text, f"{file_name} (Page {idx+1})"))
                    
        elif ext in ['docx', 'doc']:
            doc = Document(io.BytesIO(file_bytes))
            full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            chunks.extend(chunk_text(full_text, f"{file_name} (Doc Text)"))
            
        elif ext == 'json':
            data = json.loads(file_bytes.decode('utf-8'))
            data_str = json.dumps(data)
            chunks.extend(chunk_text(data_str, file_name, max_words=300))
            
        else:
             chunks.extend(chunk_text(file_bytes.decode('utf-8', errors='ignore'), file_name))
             
    except Exception as e:
        st.warning(f"⚠️ Could not parse {file_name}: {e}")
    return chunks

# --- EXECUTE INGESTION PIPELINE ---
if fetch_btn:
    with st.status("🚀 Processing Data Pipeline...", expanded=True) as status:
        st.write("📡 Fetching directory from Google Drive...")
        files = fetch_files_from_drive(DRIVE_FOLDER_ID, GOOGLE_DRIVE_API_KEY)
        
        if not files:
            status.update(label="❌ No files found in folder. Check Folder ID and Permissions.", state="error")
            st.stop()
            
        all_chunks = []
        processed_names = []
        
        for f in files:
            f_name, f_id = f['name'], f['id']
            st.write(f"📥 Downloading: {f_name}...")
            
            f_bytes, error_msg = download_drive_file(f_id, GOOGLE_DRIVE_API_KEY)
            
            if f_bytes:
                st.write(f"⚙️ Extracting data from: {f_name}")
                file_chunks = parse_file_content(f_name, f_bytes)
                if file_chunks:
                    all_chunks.extend(file_chunks)
                    processed_names.append(f_name)
            else:
                st.error(f"❌ Failed to download {f_name}: {error_msg}")
        
        if not all_chunks:
            status.update(label="❌ Pipeline Failed: No valid data could be extracted.", state="error")
            st.stop()
            
        st.write(f"🧠 Generating AI Embeddings for {len(all_chunks)} chunks...")
        progress_bar = st.progress(0)
        
        st.session_state.vectorstore = []
        
        # Determine Embedding Model dynamically
        embed_model = "models/text-embedding-004"
        try:
            valid = [m.name for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
            if valid: embed_model = valid[0]
        except Exception:
            pass

        # Robust embedding with fallback logic
        for i, chunk in enumerate(all_chunks):
            for attempt in range(3):
                try:
                    emb = genai.embed_content(
                        model=embed_model,
                        content=chunk['text'],
                        task_type="retrieval_document"
                    )['embedding']
                    
                    st.session_state.vectorstore.append({
                        "vector": emb,
                        "text": chunk['text'],
                        "source": chunk['source']
                    })
                    break
                except Exception:
                    time.sleep(2) # 2 sec backoff to prevent API quota crash
            
            progress_bar.progress((i + 1) / len(all_chunks))
            
        if not st.session_state.vectorstore:
            status.update(label="❌ Failed to generate embeddings. Check Gemini API key limit.", state="error")
            st.stop()
            
        st.session_state.processed_files = processed_names
        status.update(label="✅ Ingestion Complete! Chatbot is armed and ready.", state="complete")
        time.sleep(1)
        st.rerun()

# --- CHAT INTERFACE & ENGINE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("html_chart"):
            components.html(message["html_chart"], height=500)

if user_query := st.chat_input("Ask DataIntern about your business logs, e.g., 'Chart revenue by region'..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    if st.session_state.vectorstore:
        embed_model = "models/text-embedding-004"
        try:
            valid = [m.name for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
            if valid: embed_model = valid[0]
        except Exception: pass

        query_embedding = None
        for attempt in range(3):
            try:
                query_embedding = genai.embed_content(
                    model=embed_model,
                    content=user_query,
                    task_type="retrieval_query"
                )['embedding']
                break
            except Exception:
                time.sleep(1.5)
                
        if not query_embedding:
            st.error("AI API is currently busy. Please wait a moment and try asking again.")
            st.stop()
        
        # Rank by pure Cosine distance
        scored = [(cosine_similarity(query_embedding, item["vector"]), item) for item in st.session_state.vectorstore]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = scored[:20] # Provide top 20 blocks for maximum context
        
        context_blocks = [f"Source [{item['source']}]: {item['text']}\n" for score, item in top_k]
        context_str = "\n".join(context_blocks)
        
        system_prompt = f"""
        You are DataIntern, a RAG engine.
        1. Answer based ONLY on the context. If not found, say "I don't see that in your files."
        2. Format output strictly as JSON.

        If text answer:
        {{
            "requires_chart": false,
            "text_response": "Your factual answer here.",
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
            with st.spinner("Analyzing business intelligence..."):
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content([system_prompt, user_query])
                    
                    # Bulletproof JSON extraction
                    raw_response = response.text.strip()
                    json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                    
                    if json_match:
                        response_data = json.loads(json_match.group(0))
                    else:
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
                    st.error(f"Failed to parse AI output. The query might be too complex or data formatting is irregular. {e}")
    elif not st.session_state.vectorstore:
        st.info("Please ingest a Google Drive folder in the sidebar to start.")
