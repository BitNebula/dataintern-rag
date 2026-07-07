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
if "auto_start" not in st.session_state:
    st.session_state.auto_start = False

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration Panel")
    st.success("✅ Secure AI Core Initialized.")
    fetch_btn = st.button("Refresh Files From Drive", type="primary")

    st.markdown("---")
    st.markdown("### 📋 System Status")
    if st.session_state.processed_files:
        st.success(f"Ingested {len(st.session_state.processed_files)} files.")
        for f in st.session_state.processed_files:
            st.text(f"• {f}")
    else:
        st.info("No documents currently loaded into Vector Store.")

# --- HELPERS ---
def get_best_model(method='generateContent'):
    """Dynamically find a model supporting the required method."""
    try:
        models = [m.name for m in genai.list_models() if method in m.supported_generation_methods]
        for preferred in ['models/gemini-1.5-flash', 'models/gemini-pro', 'gemini-pro']:
            if preferred in models: return preferred
        return models[0] if models else 'gemini-pro'
    except:
        return 'gemini-pro'

def cosine_similarity(a, b):
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0: return 0.0
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
    url_web = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        req = urllib.request.Request(url_web, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.read(), None
    except Exception:
        url_api = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
        try:
            req = urllib.request.Request(url_api, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=20) as response:
                return response.read(), None
        except urllib.error.HTTPError as e:
            return None, f"HTTP {e.code}: Could not fetch bytes."
        except Exception as e:
            return None, str(e)

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
                if text and text.strip(): chunks.extend(chunk_text(text, f"{file_name} (Page {idx+1})"))
        elif ext in ['docx', 'doc']:
            doc = Document(io.BytesIO(file_bytes))
            full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            chunks.extend(chunk_text(full_text, f"{file_name} (Doc Text)"))
        elif ext == 'json':
            data = json.loads(file_bytes.decode('utf-8'))
            chunks.extend(chunk_text(json.dumps(data), file_name, max_words=300))
        else:
             chunks.extend(chunk_text(file_bytes.decode('utf-8', errors='ignore'), file_name))
    except Exception as e:
        st.warning(f"⚠️ Could not parse {file_name}: {e}")
    return chunks

# --- PIPELINE ---
if fetch_btn or not st.session_state.auto_start:
    st.session_state.auto_start = True
    with st.status("🚀 Processing Data Pipeline...", expanded=True) as status:
        st.write("📡 Fetching directory...")
        files = fetch_files_from_drive(DRIVE_FOLDER_ID, GOOGLE_DRIVE_API_KEY)
        if not files:
            status.update(label="❌ No files found.", state="error")
            st.stop()
            
        all_chunks = []
        processed_names = []
        for f in files:
            f_name, f_id = f['name'], f['id']
            st.write(f"📥 Downloading: {f_name}...")
            f_bytes, error_msg = download_drive_file(f_id, GOOGLE_DRIVE_API_KEY)
            if f_bytes:
                st.write(f"⚙️ Parsing: {f_name}")
                chunks = parse_file_content(f_name, f_bytes)
                if chunks:
                    all_chunks.extend(chunks)
                    processed_names.append(f_name)
            else:
                st.error(f"❌ Failed to download {f_name}: {error_msg}")
        
        st.write(f"🧠 Generating Embeddings...")
        st.session_state.vectorstore = []
        embed_model = get_best_model(method='embedContent')
        
        for i, chunk in enumerate(all_chunks):
            try:
                emb = genai.embed_content(model=embed_model, content=chunk['text'], task_type="retrieval_document")['embedding']
                st.session_state.vectorstore.append({"vector": emb, "text": chunk['text'], "source": chunk['source']})
            except: pass
            
        st.session_state.processed_files = processed_names
        status.update(label="✅ Ingestion Complete!", state="complete")
        time.sleep(1)
        st.rerun()

# --- CHAT ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("html_chart"): components.html(message["html_chart"], height=500)

if user_query := st.chat_input("Ask DataIntern to chart your data..."):
    with st.chat_message("user"): st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    if st.session_state.vectorstore:
        embed_model = get_best_model(method='embedContent')
        query_emb = genai.embed_content(model=embed_model, content=user_query, task_type="retrieval_query")['embedding']
        
        scored = sorted([(cosine_similarity(query_emb, i["vector"]), i) for i in st.session_state.vectorstore], key=lambda x: x[0], reverse=True)[:20]
        context = "\n".join([f"[{i['source']}]: {i['text']}" for s, i in scored])
        
        system_prompt = f"""
        You are DataIntern, a strict RAG data analyst.
        CRITICAL INSTRUCTION: You MUST format your entire response as a single, valid JSON object. NEVER output plain conversational text.
        
        If the user asks a plain text question, use this exact format:
        {{
            "requires_chart": false,
            "text_response": "Factual answer based on context."
        }}

        If the user asks for a chart, graph, or says "anything relevant", you MUST extract and aggregate numerical data from the context (for example, sum up amounts by Owner, or count Deals by Stage) and use this exact format:
        {{
            "requires_chart": true,
            "text_response": "Here is the visualized data:",
            "chart_data": {{
                "type": "bar",
                "title": "Descriptive Chart Title",
                "x_label": "X Axis Label",
                "y_label": "Y Axis Label",
                "x_data": ["Category A", "Category B", "Category C"],
                "y_data": [10, 50, 25]
            }}
        }}

        CONTEXT DATA:
        {context}
        """

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data and generating insights..."):
                try:
                    # DYNAMIC MODEL SELECTION: Automatically grab the best available chat model for your specific API key
                    chat_model_name = get_best_model(method='generateContent')
                    model = genai.GenerativeModel(chat_model_name)
                    response = model.generate_content([system_prompt, f"User Query: {user_query}"])
                    
                    # BULLETPROOF JSON PARSING
                    raw = response.text.strip()
                    raw = raw.replace('```json', '').replace('```', '').strip()
                    
                    try:
                        res = json.loads(raw)
                    except Exception:
                        # If raw parse fails, use Regex rescue
                        match = re.search(r'\{.*\}', raw, re.DOTALL)
                        if match:
                            res = json.loads(match.group(0))
                        else:
                            res = {"requires_chart": False, "text_response": raw} # Emergency text fallback
                    
                    # RENDER RESPONSE
                    st.markdown(res.get("text_response", "Here are your insights:"))
                    
                    if res.get("requires_chart") and res.get("chart_data"):
                        c = res.get("chart_data", {})
                        fig = go.Figure()
                        ctype = c.get('type', 'bar')
                        x_val, y_val = c.get('x_data', []), c.get('y_data', [])
                        
                        if ctype == 'bar': fig.add_trace(go.Bar(x=x_val, y=y_val))
                        elif ctype == 'line': fig.add_trace(go.Scatter(x=x_val, y=y_val, mode='lines+markers'))
                        elif ctype == 'pie': fig.add_trace(go.Pie(labels=x_val, values=y_val))
                        elif ctype == 'scatter': fig.add_trace(go.Scatter(x=x_val, y=y_val, mode='markers'))
                        
                        fig.update_layout(title=c.get('title', 'Data Insights'), xaxis_title=c.get('x_label', ''), yaxis_title=c.get('y_label', ''))
                        html_chart = fig.to_html(full_html=True, include_plotlyjs='cdn')
                        
                        st.components.v1.html(html_chart, height=500)
                        st.session_state.messages.append({"role": "assistant", "content": res.get("text_response"), "html_chart": html_chart})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": res.get("text_response")})
                        
                except Exception as e:
                    st.error(f"Analysis failed: Could not render AI response. Error details: {e}")
