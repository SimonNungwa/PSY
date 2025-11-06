import os
import requests
import streamlit as st
import re


st.set_page_config(page_title="Your AI Friend", page_icon="💬", layout="centered")
st.title("💬 Chat with Your AI Friend")

API_URL = "https://router.huggingface.co/v1/chat/completions"

if "HF_TOKEN" not in os.environ:
    st.error("Missing HF_TOKEN environment variable.")
    st.info("Run this in terminal before starting Streamlit:\n\n`export HF_TOKEN='hf_your_token_here'`")
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
    "Content-Type": "application/json",
}

# Friendly system prompt
FRIEND_PROMPT = """You are a warm, supportive, and engaging friend. Your conversation style is:
- Natural and conversational, like texting a close friend
- Empathetic and genuinely interested in the person you're talking to
- Positive and encouraging, but authentic (not overly enthusiastic)
- Use casual language, but stay articulate and thoughtful
- Share relatable thoughts and perspectives
- Ask follow-up questions to show you care
- Use occasional emojis naturally when it fits the mood
- Be a good listener and remember context from the conversation
- Offer advice when asked, but mostly be there to chat and connect

Keep responses concise and engaging. Break up longer thoughts into shorter paragraphs for easy reading."""


def format_response(text):
    """Format the AI response for better readability."""
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    
    return text.strip()


def query_deepseek(messages, model="deepseek-ai/DeepSeek-R1:novita"):
    """Send a chat request to the Hugging Face router."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.8,  # More creative and conversational
        "max_tokens": 500,   # Keep responses concise
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        if response.status_code != 200:
            st.error(f"Error {response.status_code}: {response.text}")
            return None

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return format_response(content)
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


# Initialize chat history with friendly system prompt
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": FRIEND_PROMPT}
    ]


# Sidebar settings
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.text_input("Model:", "deepseek-ai/DeepSeek-R1:novita")

if st.sidebar.button("🔄 Start New Conversation"):
    st.session_state.messages = [
        {"role": "system", "content": FRIEND_PROMPT}
    ]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("""
- Share what's on your mind
- Ask for advice or just chat
- Be yourself!
""")


# Display chat messages (skip system message)
for msg in st.session_state.messages[1:]:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("What's on your mind?")

if user_input:
    # Display user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get and display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            reply = query_deepseek(st.session_state.messages, model_name)
        
        if reply:
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            st.error("Sorry, I couldn't get a response. Please try again!")

# Footer
st.markdown("---")
st.caption(f"🤖 Powered by {model_name}")