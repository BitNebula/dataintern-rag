import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import numpy as np
import plotly.express as px
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
if "vector_db" not in st.session_state:
    st.session_state.vector_db = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
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
        st.success(f"Ingested {len(st.session_state.processed_files)} files successfully.")
        for f in st.session_state.processed_files:
            st.text(f"• {f}")
    else:
        st.info("No documents currently loaded into Vector Store.")

# --- HELPERS: PURE PYTHON VECTOR MATH ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- GOOGLE DRIVE FILE INGESTION VIA API KEY ---
def fetch_files_from_drive(folder_id, api_key):
    url = f"https://www.googleapis.com/drive/v3/files?q='{folder_id}'+in+parents&key={api_key}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data.get('files', [])
    except Exception as e:
        st.sidebar.error(f"Failed to access Google Drive: {e}")
        return []

def download_drive_file(file_id, api_key):
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            return response.read()
    except Exception as e:
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
                    chunks.append({"text": f"Sheet: {sheet} | Row {idx}: {row_str}", "source": f"{file_name} (Sheet: {sheet}, Row {idx})"})
                    
        elif ext == 'pdf':
            pdf_reader = PdfReader(io.BytesIO(file_bytes))
            for idx, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    chunks.append({"text": text, "source": f"{file_name} (Page {idx+1})"})
                    
        elif ext in ['docx', 'doc']:
            doc = Document(io.BytesIO(file_bytes))
            full_text = [para.text for para in doc.paragraphs if para.text.strip()]
            for idx, para in enumerate(full_text):
                chunks.append({"text": para, "source": f"{file_name} (Paragraph {idx+1})"})
                
        elif ext == 'json':
            data = json.loads(file_bytes.decode('utf-8'))
            data_str = json.dumps(data, indent=2)
            chunks.append({"text": data_str, "source": file_name})
            
    except Exception as e:
        st.warning(f"Could not parse file {file_name}: {e}")
        
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
                f_name = f['name']
                f_id = f['id']
                f_bytes = download_drive_file(f_id, GOOGLE_DRIVE_API_KEY)
                
                if f_bytes:
                    file_chunks = parse_file_content(f_name, f_bytes)
                    all_chunks.extend(file_chunks)
                    processed_names.append(f_name)
            
            if all_chunks:
                with st.spinner("Generating embeddings and building vector catalog..."):
                    texts = [c['text'] for c in all_chunks]
                    response = genai.embed_content(
                        model="models/text-embedding-004",
                        content=texts,
                        task_type="retrieval_document"
                    )
                    
                    st.session_state.vector_db = []
                    for i, embedding in enumerate(response['embedding']):
                        st.session_state.vector_db.append({
                            "vector": embedding,
                            "text": all_chunks[i]['text'],
                            "source": all_chunks[i]['source']
                        })
                st.session_state.processed_files = processed_names
                st.rerun()

# --- CHAT INTERFACE & ENGINE ---
# Render historical turns
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["type"] == "text":
            st.write(message["content"])
        elif message["type"] == "chart":
            st.write(message["content"])
            fig_data = pd.DataFrame(message["chart_data"])
            if message["chart_type"] == "bar":
                st.plotly_chart(px.bar(fig_data, x="label", y="value", title=message["chart_title"]))
            elif message["chart_type"] == "line":
                st.plotly_chart(px.plotly_chart(px.line(fig_data, x="label", y="value", title=message["chart_title"])))
            elif message["chart_type"] == "pie":
                st.plotly_chart(px.pie(fig_data, names="label", values="value", title=message["chart_title"]))
            elif message["chart_type"] == "scatter":
                st.plotly_chart(px.scatter(fig_data, x="label", y="value", title=message["chart_title"]))

# Handle user interaction
if user_query := st.chat_input("Ask DataIntern about your business logs, performance tracking, or custom charts..."):
    with st.chat_message("user"):
        st.write(user_query)
    
    st.session_state.chat_history.append({"role": "user", "type": "text", "content": user_query})
    
    # --- RAG RETRIEVAL PIPELINE ---
    context_str = ""
    if st.session_state.vector_db:
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=user_query,
            task_type="retrieval_query"
        )['embedding']
        
        scored_chunks = [(cosine_similarity(query_embedding, item["vector"]), item) for item in st.session_state.vector_db]
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_chunks[:15]
        
        context_blocks = [f"Source [{item['source']}]: {item['text']}\n" for score, item in top_k]
        context_str = "\n".join(context_blocks)
    
    system_prompt = f"""
    You are DataIntern, an advanced full-stack RAG engine designed to interpret tabular, financial, and multi-format business logs.
    
    STRICT GROUNDING DIRECTIVES:
    1. Answer the query based ONLY on the Provided Context below.
    2. If the answer cannot be confidently deduced from the text context, respond exactly with: "I don't see that in your files." Do not try to make up information.
    3. When providing answers, append the Source tags precisely as mentioned in the context blocks.
    
    OUTPUT FORMAT SPECIFICATIONS:
    Your response must be a single parseable JSON block matching one of these two structures:

    For text answers:
    {{
        "type": "text",
        "content": "Your factual text response here incorporating source citations."
    }}

    For chart/graph generation requests:
    {{
        "type": "chart",
        "content": "Short textual summary of data insights shown in the chart.",
        "chart_type": "bar", // or "line", "pie", "scatter"
        "chart_title": "Descriptive Chart Title detailing source parameters",
        "chart_data": [
            {{"label": "X-axis item string", "value": 12500.50}},
            {{"label": "Next item string", "value": 14200.00}}
        ]
    }}

    PROVIDED DATA CONTEXT:
    {context_str if context_str else "No files have been loaded."}
    """
    
    with st.chat_message("assistant"):
        with st.spinner("Processing deep queries..."):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(
                    [system_prompt, f"User Query: {user_query}"],
                    generation_config={"response_mime_type": "application/json"}
                )
                
                res_payload = json.loads(response.text)
                
                if res_payload["type"] == "text":
                    st.write(res_payload["content"])
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "type": "text",
                        "content": res_payload["content"]
                    })
                elif res_payload["type"] == "chart":
                    st.write(res_payload["content"])
                    fig_data = pd.DataFrame(res_payload["chart_data"])
                    
                    if res_payload["chart_type"] == "bar":
                        st.plotly_chart(px.bar(fig_data, x="label", y="value", title=res_payload["chart_title"]))
                    elif res_payload["chart_type"] == "line":
                        st.plotly_chart(px.line(fig_data, x="label", y="value", title=res_payload["chart_title"]))
                    elif res_payload["chart_type"] == "pie":
                        st.plotly_chart(px.pie(fig_data, names="label", values="value", title=res_payload["chart_title"]))
                    elif res_payload["chart_type"] == "scatter":
                        st.plotly_chart(px.scatter(fig_data, x="label", y="value", title=res_payload["chart_title"]))
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "type": "chart",
                        "content": res_payload["content"],
                        "chart_type": res_payload["chart_type"],
                        "chart_title": res_payload["chart_title"],
                        "chart_data": res_payload["chart_data"]
                    })
            except Exception as ex:
                st.error(f"Error executing engine request: {ex}")
