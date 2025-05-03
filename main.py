import os
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import requests
import chardet

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory

# ─── Configuration ─────────────────────────────────────────────────────────────
# First, try Streamlit secrets, fallback to environment variables
try:
    API_BASE        = st.secrets["api"]["base_url"]
    USERNAME        = st.secrets["api"]["username"]
    PASSWORD        = st.secrets["api"]["password"]
    OPENAI_API_KEY  = st.secrets["api"]["openai_api_key"]
except Exception:
    API_BASE        = os.getenv("API_BASE")
    USERNAME        = os.getenv("API_USERNAME")
    PASSWORD        = os.getenv("API_PASSWORD")
    OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

# Validate configuration
if not all([API_BASE, USERNAME, PASSWORD, OPENAI_API_KEY]):
    st.error("API credentials or secrets are not set. Please configure them via Streamlit secrets or environment variables.")
    st.stop()


# ─── Helper Functions ──────────────────────────────────────────────────────────

def get_access_token():
    resp = requests.post(
        f"{API_BASE}/auth/login",
        data={"username": USERNAME, "password": PASSWORD, "remember": "true"}
    )
    resp.raise_for_status()
    return resp.json()["content"]["access_token"]


def detect_encoding(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return chardet.detect(f.read()).get("encoding", "utf-8")


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # parse dates/times
    for col in df.columns:
        if any(k in col.lower() for k in ("date","time","day","month","year")):
            try:
                df[col] = pd.to_datetime(df[col], dayfirst=True)
            except Exception:
                pass
    # convert numeric-like strings
    for col in df.select_dtypes(include="object"):
        sample = df[col].dropna().astype(str).head(5)
        if all(s.replace('.', '', 1).replace('-', '', 1).isdigit() for s in sample):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ─── App Setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📊 Chat with CSV – Maids.cc", layout="wide")

# initialize session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()
if "df" not in st.session_state:
    st.session_state.df = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "token" not in st.session_state:
    st.session_state.token = None

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")
    source = st.radio("Data source", ["Upload CSV", "API Call"])
    if source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    else:
        agents          = st.text_input("Agent")
        queues          = st.text_input("Queues")
        directions      = st.multiselect("Direction", ["inbound","outbound","eavesdrop"], default=[])
        from_date_picker= st.date_input("From date")
        from_time_picker= st.time_input("From time")
        to_date_picker  = st.date_input("To date")
        to_time_picker  = st.time_input("To time")
        call_id         = st.text_input("Call ID")
        result          = st.multiselect("Result", ["answered", "cancel", "no-answer", "failed", "busy", "lose-race", "timeout", "transfer"])
    load_btn = st.button("Load Data")

# ─── Data Loading ───────────────────────────────────────────────────────────────
st.title("📊 Chat with CSV – Maids.cc")
if source == "Upload CSV" and uploaded_file and load_btn:
    tmp = "temp.csv"
    with open(tmp, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.session_state.df = pd.read_csv(tmp, encoding=detect_encoding(tmp))
    os.remove(tmp)
elif source == "API Call" and load_btn:
    st.session_state.token = get_access_token()
    params = []
    # multi-value filters
    for a in agents:
        params.append(("agentName", a))
    for q in queues:
        params.append(("queues", q))
    for d in directions:
        params.append(("direction", d))
    # single-value filters

    # combined date & time
    if from_date_picker:
        params.append(("fromDate", from_date_picker.strftime("%Y-%m-%d")))
    if from_time_picker:
        params.append(("fromTime", from_time_picker.strftime("%H:%M")))
    if to_date_picker:
        params.append(("toDate", to_date_picker.strftime("%Y-%m-%d")))
    if to_time_picker:
        params.append(("toTime", to_time_picker.strftime("%H:%M")))
    if call_id:
        params.append(("callID", call_id))
    if result:
        params.append(("result", result))
    # call API
    resp = requests.get(
        f"{API_BASE}/callHistory/export/csv/", params=params,
        headers={"access_token": st.session_state.token}
    )
    resp.raise_for_status()
    st.session_state.df = pd.read_csv(io.StringIO(resp.text))
    st.success(f"Loaded {len(st.session_state.df)} rows")

if st.session_state.df is None:
    st.info("Click 'Load Data' after selecting options.")
    st.stop()

# ─── Display & Preprocess ───────────────────────────────────────────────────────
df = preprocess_dataframe(st.session_state.df)
st.write(f"DataFrame: {df.shape[0]} rows × {df.shape[1]} cols")
st.dataframe(df)

# ─── Initialize Agent Once ─────────────────────────────────────────────────────
if st.session_state.agent is None:
    st.session_state.agent = create_pandas_dataframe_agent(
        llm=ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.1,
            openai_api_key=OPENAI_API_KEY
        ),
        df=df,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

# ─── Chat Interface ────────────────────────────────────────────────────────────
st.subheader("Ask questions about your data")
user_input = st.chat_input("Type your question and press Enter...")
if user_input:
    with st.spinner("Thinking..."):
        answer = st.session_state.agent.run(
            input=user_input,
            history=st.session_state.memory.buffer
        )
        st.session_state.memory.save_context({"input": user_input}, {"output": answer})
        st.session_state.messages.append({"role":"user","content":user_input})
        st.session_state.messages.append({"role":"assistant","content":answer})
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
