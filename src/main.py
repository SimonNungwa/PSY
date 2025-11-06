import os
import requests
import streamlit as st


st.set_page_config(page_title="DeepSeek Chatbot", layout="centered")
st.title("Chatbot")

API_URL = "https://router.huggingface.co/v1/chat/completions"

if "HF_TOKEN" not in os.environ:
    st.error("Missing HF_TOKEN environment variable.")
    st.info("Run this in terminal before starting Streamlit:\n\n`export HF_TOKEN='hf_your_token_here'`")
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
    "Content-Type": "application/json",
}


def query_deepseek(messages, model="deepseek-ai/DeepSeek-R1:novita"):
    """Send a chat request to the Hugging Face router."""
    payload = {
        "model": model,
        "messages": messages,
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        if response.status_code != 200:
            st.error(f"Error {response.status_code}: {response.text}")
            return None

        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None



if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]


st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.text_input("Model:", "deepseek-ai/DeepSeek-R1:novita")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]
    st.rerun()


for msg in st.session_state.messages[1:]:
    speaker = "" if msg["role"] == "user" else " "
    st.markdown(f"**{speaker}:** {msg['content']}")

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        reply = query_deepseek(st.session_state.messages, model_name)
    if reply:
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

st.markdown("---")
st.caption(f" Model: {model_name}")
