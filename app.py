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

# Load Secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GOOGLE_DRIVE_API_KEY = st.secrets["GOOGLE_DRIVE_API_KEY"]
    DRIVE_FOLDER_ID = st.secrets["DRIVE_FOLDER_ID"]
except KeyError as e:
    st.error(f"Missing Secret: {e}. Please add it to your Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# Initialize State
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# --- CORE MATH & PARSING ---
def cosine_similarity(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if norm == 0 else np.dot(a, b) / norm

def chunk_text(text, source_name, max_words=600):
    words = str(text).split()
    return [{"text": " ".join(words[i:i+max_words]), "source": f"{source_name} (Part {i//max_words + 1})"} for i in range(0, len(words), max_words)]

def parse_file(file_name, file_bytes):
    chunks, ext = [], file_name.split('.')[-1].lower()
    try:
        if ext == 'csv':
            df = pd.read_csv(io.BytesIO(file_bytes))
            chunks.extend(chunk_text(df.to_csv(index=False), file_name))
        elif ext in ['xlsx', 'xls']:
            xl = pd.ExcelFile(io.BytesIO(file_bytes))
            for sheet in xl.sheet_names:
                df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
                chunks.extend(chunk_text(df.to_csv(index=False), f"{file_name} ({sheet})"))
        elif ext == 'pdf':
            pdf_reader = PdfReader(io.BytesIO(file_bytes))
            chunks.extend(chunk_text(" ".join([p.extract_text() or "" for p in pdf_reader.pages]), file_name))
        elif ext in ['docx', 'doc']:
            doc = Document(io.BytesIO(file_bytes))
            chunks.extend(chunk_text(" ".join([p.text for p in doc.paragraphs]), file_name))
        elif ext == 'json':
            chunks.extend(chunk_text(json.dumps(json.loads(file_bytes.decode('utf-8'))), file_name))
    except Exception:
        pass
    return chunks

# --- INGESTION PIPELINE ---
if not st.session_state.data_loaded:
    with st.status("🚀 Syncing with Google Drive...", expanded=True) as status:
        url = f"https://www.googleapis.com/drive/v3/files?q='{DRIVE_FOLDER_ID}'+in+parents&key={GOOGLE_DRIVE_API_KEY}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                files = json.loads(resp.read().decode()).get('files', [])
        except Exception as e:
            st.error(f"Failed to connect to Google Drive: {e}")
            st.stop()
            
        all_chunks, processed = [], []
        for f in files:
            st.write(f"📥 Pulling: {f['name']}")
            dl_url = f"https://www.googleapis.com/drive/v3/files/{f['id']}?alt=media&key={GOOGLE_DRIVE_API_KEY}"
            try:
                req = urllib.request.Request(dl_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    all_chunks.extend(parse_file(f['name'], resp.read()))
                    processed.append(f['name'])
            except Exception:
                pass
        
        st.write(f"🧠 Embedding {len(all_chunks)} chunks...")
        progress = st.progress(0)
        
        for i, chunk in enumerate(all_chunks):
            try:
                # With SDK updated, standard text-embedding-004 will connect natively
                emb = genai.embed_content(model="models/text-embedding-004", content=chunk['text'], task_type="retrieval_document")['embedding']
                st.session_state.vectorstore.append({"vector": emb, "text": chunk['text'], "source": chunk['source']})
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    time.sleep(5) # Pause briefly only if we hit a true speed limit
            progress.progress((i + 1) / len(all_chunks))
            time.sleep(1) # Steady 1-second pace keeps us well below free-tier limits
            
        st.session_state.processed_files = processed
        st.session_state.data_loaded = True
        status.update(label="✅ Ready!", state="complete")
        time.sleep(1)
        st.rerun()

# --- SIDEBAR & CHAT ---
with st.sidebar:
    st.header("⚙️ System Status")
    if st.button("Force Refresh Data", type="primary"):
        st.session_state.data_loaded = False
        st.session_state.vectorstore = []
        st.rerun()
    if st.session_state.processed_files:
        st.success(f"{len(st.session_state.processed_files)} files active.")
        for f in st.session_state.processed_files: st.text(f"• {f}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("html_chart"): components.html(msg["html_chart"], height=500)

if user_query := st.chat_input("Ask a question or request a chart..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"): st.markdown(user_query)
        
    with st.chat_message("assistant"):
        try:
            query_emb = genai.embed_content(model="models/text-embedding-004", content=user_query, task_type="retrieval_query")['embedding']
            scored = sorted([(cosine_similarity(query_emb, i["vector"]), i) for i in st.session_state.vectorstore], key=lambda x: x[0], reverse=True)[:15]
            context_str = "\n".join([f"[{i['source']}]: {i['text']}" for _, i in scored])
            
            prompt = f"""
            You are DataIntern, a data analyst. Use ONLY the provided context. Output ONLY valid JSON.
            Format:
            {{
                "requires_chart": true/false,
                "text_response": "Your answer here",
                "chart_data": {{"type": "bar", "title": "Title", "x_label": "X", "y_label": "Y", "x_data": ["A", "B"], "y_data": [10, 20]}}
            }}
            CONTEXT:\n{context_str}\n\nQUERY: {user_query}
            """
            
            model = genai.GenerativeModel("gemini-1.5-flash")
            raw_res = model.generate_content(prompt).text.strip().replace('```json', '').replace('```', '').strip()
            
            try:
                res = json.loads(raw_res)
            except Exception:
                match = re.search(r'\{.*\}', raw_res, re.DOTALL)
                res = json.loads(match.group(0)) if match else {"requires_chart": False, "text_response": raw_res}
                
            st.markdown(res.get("text_response", ""))
            
            html_chart = None
            if res.get("requires_chart") and res.get("chart_data"):
                c = res["chart_data"]
                fig = go.Figure()
                ctype, x_val, y_val = c.get('type', 'bar'), c.get('x_data', []), c.get('y_data', [])
                if ctype == 'bar': fig.add_trace(go.Bar(x=x_val, y=y_val))
                elif ctype == 'line': fig.add_trace(go.Scatter(x=x_val, y=y_val, mode='lines+markers'))
                elif ctype == 'pie': fig.add_trace(go.Pie(labels=x_val, values=y_val))
                elif ctype == 'scatter': fig.add_trace(go.Scatter(x=x_val, y=y_val, mode='markers'))
                fig.update_layout(title=c.get('title', ''), xaxis_title=c.get('x_label', ''), yaxis_title=c.get('y_label', ''))
                html_chart = fig.to_html(full_html=True, include_plotlyjs='cdn')
                components.html(html_chart, height=500)
                
            st.session_state.messages.append({"role": "assistant", "content": res.get("text_response", ""), "html_chart": html_chart})
        except Exception as e:
            st.error(f"❌ Error: {e}")
