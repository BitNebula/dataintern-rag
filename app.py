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
import time
import re

st.set_page_config(page_title="DataIntern - AI CRM", layout="wide")
st.title("💼 DataIntern: Auto-RAG & Analytics Engine")
st.caption("Secure Multi-Format Ingestion & Instant Visualization")

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GOOGLE_DRIVE_API_KEY = st.secrets["GOOGLE_DRIVE_API_KEY"]
    DRIVE_FOLDER_ID = st.secrets["DRIVE_FOLDER_ID"]
except KeyError as e:
    st.error(f"Missing Secret: {e}. Please add it to your Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

def embed_text_safe(text, task_type):
    """Uses hardcoded, proven stable models. Exposes exact error if it fails."""
    last_error = None
    for model_name in ["models/text-embedding-004", "models/embedding-001"]:
        for attempt in range(3):
            try:
                return genai.embed_content(model=model_name, content=text, task_type=task_type)['embedding']
            except Exception as e:
                last_error = e
                time.sleep(3)
    raise Exception(f"Model {model_name} failed. Error: {str(last_error)}")

def chat_safe(prompt):
    """Uses hardcoded, proven stable models. Exposes exact error if it fails."""
    last_error = None
    for model_name in ["gemini-1.5-flash", "gemini-pro"]:
        for attempt in range(3):
            try:
                model = genai.GenerativeModel(model_name)
                return model.generate_content(prompt).text
            except Exception as e:
                last_error = e
                time.sleep(3)
    raise Exception(f"Model {model_name} failed. Error: {str(last_error)}")

def cosine_similarity(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if norm == 0 else np.dot(a, b) / norm

def fetch_files_from_drive(folder_id, api_key):
    url = f"https://www.googleapis.com/drive/v3/files?q='{folder_id}'+in+parents&key={api_key}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode()).get('files', [])
    except Exception:
        return []

def download_drive_file(file_id, api_key):
    url_web = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        req = urllib.request.Request(url_web, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.read()
    except Exception:
        url_api = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
        try:
            req = urllib.request.Request(url_api, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                return response.read()
        except Exception:
            return None

def chunk_text_safe(text, source_name, max_words=800):
    chunks = []
    words = str(text).split()
    for i in range(0, len(words), max_words):
        chunks.append({"text": " ".join(words[i:i+max_words]), "source": f"{source_name} (Part {i//max_words + 1})"})
    return chunks

def parse_file_content(file_name, file_bytes):
    chunks = []
    ext = file_name.split('.')[-1].lower()
    try:
        if ext == 'csv':
            df = pd.read_csv(io.BytesIO(file_bytes))
            chunks.extend(chunk_text_safe(df.to_csv(index=False), file_name))
        elif ext in ['xlsx', 'xls']:
            xl = pd.ExcelFile(io.BytesIO(file_bytes))
            for sheet in xl.sheet_names:
                df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
                chunks.extend(chunk_text_safe(df.to_csv(index=False), f"{file_name} ({sheet})"))
        elif ext == 'pdf':
            pdf_reader = PdfReader(io.BytesIO(file_bytes))
            full_text = " ".join([page.extract_text() or "" for page in pdf_reader.pages])
            chunks.extend(chunk_text_safe(full_text, file_name))
        elif ext in ['docx', 'doc']:
            doc = Document(io.BytesIO(file_bytes))
            full_text = " ".join([p.text for p in doc.paragraphs])
            chunks.extend(chunk_text_safe(full_text, file_name))
        elif ext == 'json':
            data = json.loads(file_bytes.decode('utf-8'))
            chunks.extend(chunk_text_safe(json.dumps(data), file_name))
    except Exception:
        pass
    return chunks

if not st.session_state.data_loaded:
    with st.status("🚀 Automatically syncing securely with Google Drive...", expanded=True) as status:
        st.write("📡 Connecting to Drive...")
        files = fetch_files_from_drive(DRIVE_FOLDER_ID, GOOGLE_DRIVE_API_KEY)
        
        if files:
            all_chunks, processed = [], []
            for f in files:
                st.write(f"📥 Pulling: {f['name']}")
                f_bytes = download_drive_file(f['id'], GOOGLE_DRIVE_API_KEY)
                if f_bytes:
                    all_chunks.extend(parse_file_content(f['name'], f_bytes))
                    processed.append(f['name'])
            
            st.write(f"🧠 Generating AI Memory for {len(all_chunks)} data chunks...")
            progress_bar = st.progress(0)
            
            for i, chunk in enumerate(all_chunks):
                try:
                    emb = embed_text_safe(chunk['text'], "retrieval_document")
                    st.session_state.vectorstore.append({"vector": emb, "text": chunk['text'], "source": chunk['source']})
                except Exception:
                    pass # Silently skip chunk if it completely fails
                
                progress_bar.progress((i + 1) / len(all_chunks))
                time.sleep(3) # Steady pace to avoid rate limits
            
            st.session_state.processed_files = processed
            
        st.session_state.data_loaded = True
        status.update(label="✅ Data Synced and System Ready!", state="complete")
        time.sleep(1)
        st.rerun()

with st.sidebar:
    st.header("⚙️ System Core")
    if st.button("Force Refresh Data", type="primary"):
        st.session_state.data_loaded = False
        st.session_state.vectorstore = []
        st.rerun()
    
    st.markdown("### 📋 Active Files")
    if st.session_state.processed_files:
        st.success(f"{len(st.session_state.processed_files)} files ingested.")
        for f in st.session_state.processed_files:
            st.text(f"• {f}")
    else:
        st.warning("No files found.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("html_chart"):
            components.html(message["html_chart"], height=500)

if user_query := st.chat_input("E.g., 'Chart the pipeline amounts by owner'"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        status_box = st.empty()
        status_box.info("🧠 Searching database & analyzing data...")
        
        try:
            query_emb = embed_text_safe(user_query, "retrieval_query")
                
            scored = [(cosine_similarity(query_emb, item["vector"]), item) for item in st.session_state.vectorstore]
            scored.sort(key=lambda x: x[0], reverse=True)
            context_str = "\n".join([f"[{i['source']}]: {i['text']}" for score, i in scored[:15]])
            
            prompt = f"""
            You are DataIntern, a strict data analyst. Use ONLY the provided context.
            Output ONLY raw JSON. No markdown ticks, no greeting.
            
            Format:
            {{
                "requires_chart": true or false,
                "text_response": "Explanation here",
                "chart_data": {{
                    "type": "bar",
                    "title": "Chart Title",
                    "x_label": "X Axis",
                    "y_label": "Y Axis",
                    "x_data": ["A", "B"],
                    "y_data": [10, 20]
                }}
            }}
            CONTEXT:
            {context_str}
            
            USER QUERY: {user_query}
            """
            
            status_box.info("📊 Generating visualization and insights...")
            raw_response = chat_safe(prompt)
            
            raw = raw_response.strip().replace('```json', '').replace('```', '').strip()
            try:
                res = json.loads(raw)
            except Exception:
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match: res = json.loads(match.group(0))
                else: res = {"requires_chart": False, "text_response": raw_response}
                
            status_box.empty()
            st.markdown(res.get("text_response", "Here is what I found:"))
            
            html_chart = None
            if res.get("requires_chart") and res.get("chart_data"):
                c = res.get("chart_data")
                fig = go.Figure()
                ctype, x_val, y_val = c.get('type', 'bar'), c.get('x_data', []), c.get('y_data', [])
                
                if ctype == 'bar': fig.add_trace(go.Bar(x=x_val, y=y_val))
                elif ctype == 'line': fig.add_trace(go.Scatter(x=x_val, y=y_val, mode='lines+markers'))
                elif ctype == 'pie': fig.add_trace(go.Pie(labels=x_val, values=y_val))
                elif ctype == 'scatter': fig.add_trace(go.Scatter(x=x_val, y=y_val, mode='markers'))
                
                fig.update_layout(title=c.get('title', ''), xaxis_title=c.get('x_label', ''), yaxis_title=c.get('y_label', ''))
                html_chart = fig.to_html(full_html=True, include_plotlyjs='cdn')
                components.html(html_chart, height=500)
                
            st.session_state.messages.append({
                "role": "assistant", 
                "content": res.get("text_response"), 
                "html_chart": html_chart
            })
            
        except Exception as e:
            status_box.empty()
            st.error(f"❌ Operation Failed: {str(e)}")
