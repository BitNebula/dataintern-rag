import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
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
if "master_context" not in st.session_state:
    st.session_state.master_context = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# --- CORE PARSING ---
def parse_file_to_text(file_name, file_bytes):
    ext = file_name.split('.')[-1].lower()
    try:
        if ext == 'csv':
            df = pd.read_csv(io.BytesIO(file_bytes))
            return df.to_csv(index=False)
        elif ext in ['xlsx', 'xls']:
            xl = pd.ExcelFile(io.BytesIO(file_bytes))
            sheets_text = []
            for sheet in xl.sheet_names:
                df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
                sheets_text.append(f"--- Sheet: {sheet} ---\n" + df.to_csv(index=False))
            return "\n".join(sheets_text)
        elif ext == 'pdf':
            pdf_reader = PdfReader(io.BytesIO(file_bytes))
            return " ".join([p.extract_text() or "" for p in pdf_reader.pages])
        elif ext in ['docx', 'doc']:
            doc = Document(io.BytesIO(file_bytes))
            return " ".join([p.text for p in doc.paragraphs])
        elif ext == 'json':
            return json.dumps(json.loads(file_bytes.decode('utf-8')), indent=2)
    except Exception:
        pass
    return ""

# --- INGESTION PIPELINE (DIRECT CONTEXT) ---
if not st.session_state.data_loaded:
    with st.status("🚀 Syncing directly with Google Drive...", expanded=True) as status:
        url = f"https://www.googleapis.com/drive/v3/files?q='{DRIVE_FOLDER_ID}'+in+parents&key={GOOGLE_DRIVE_API_KEY}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                files = json.loads(resp.read().decode()).get('files', [])
        except Exception as e:
            st.error(f"Failed to connect to Google Drive: {e}")
            st.stop()
            
        full_text_corpus = ""
        processed = []
        
        for f in files:
            st.write(f"📥 Pulling & Reading: {f['name']}")
            dl_url = f"https://www.googleapis.com/drive/v3/files/{f['id']}?alt=media&key={GOOGLE_DRIVE_API_KEY}"
            try:
                req = urllib.request.Request(dl_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    file_text = parse_file_to_text(f['name'], resp.read())
                    if file_text:
                        full_text_corpus += f"\n\n=== START SOURCE FILE: {f['name']} ===\n{file_text}\n=== END SOURCE FILE: {f['name']} ===\n"
                        processed.append(f['name'])
            except Exception:
                pass
                
        st.session_state.master_context = full_text_corpus
        st.session_state.processed_files = processed
        st.session_state.data_loaded = True
        status.update(label="✅ Context Ingested Successfully!", state="complete")
        time.sleep(1)
        st.rerun()

# --- SIDEBAR & CHAT ---
with st.sidebar:
    st.header("⚙️ System Status")
    if st.button("Force Refresh Data", type="primary"):
        st.session_state.data_loaded = False
        st.session_state.master_context = ""
        st.rerun()
    if st.session_state.processed_files:
        st.success(f"{len(st.session_state.processed_files)} files active in memory.")
        for f in st.session_state.processed_files: 
            st.text(f"• {f}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("html_chart"): 
            components.html(msg["html_chart"], height=500)

if user_query := st.chat_input("Ask a question or request a chart..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"): 
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        status_box = st.empty()
        status_box.info("🧠 Processing complete dataset context...")
        
        prompt = f"""
        You are DataIntern, a data analyst. Use ONLY the provided context. Output ONLY valid JSON.
        Format:
        {{
            "requires_chart": true/false,
            "text_response": "Your answer here",
            "chart_data": {{"type": "bar", "title": "Title", "x_label": "X", "y_label": "Y", "x_data": ["A", "B"], "y_data": [10, 20]}}
        }}
        CONTEXT:
        {st.session_state.master_context}
        
        QUERY: {user_query}
        """
        
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            raw_res = model.generate_content(prompt).text.strip().replace('```json', '').replace('```', '').strip()
            
            try:
                res = json.loads(raw_res)
            except Exception:
                match = re.search(r'\{.*\}', raw_res, re.DOTALL)
                res = json.loads(match.group(0)) if match else {"requires_chart": False, "text_response": raw_res}
                
            status_box.empty()
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
            status_box.empty()
            st.error(f"❌ Error during analysis: {e}")
