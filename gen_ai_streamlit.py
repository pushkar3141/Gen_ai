import uuid
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Pushkar's AI App", page_icon="🤖", layout="wide")

api_key = st.secrets.get("NVIDIA_API_KEY")
if not api_key:
    st.error("Add NVIDIA_API_KEY to .streamlit/secrets.toml")
    st.stop()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
)

MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "deepseek-ai/deepseek-v4-pro-0813",
    "moonshotai/kimi-k3",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
]

# ---- session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ---- sidebar ----
with st.sidebar:
    st.title("⚙️ Settings")
    model = st.selectbox("Model", MODELS, index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.05)
    max_tokens = st.slider("Max tokens", 256, 8192, 2048, 256)
    thinking = st.toggle("Thinking mode", value=False,
                         help="Enable for reasoning models (Nemotron 3, DeepSeek V4)")
    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful AI assistant.",
        height=100,
    )
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}`")

# ---- main ----
st.title("🤖 Pushkar's AI App")
st.caption(f"Powered by NVIDIA NIM · Model: `{model}`")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        reasoning_box = st.empty()
        placeholder = st.empty()
        full_response = ""
        reasoning_buffer = ""

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_tokens,
                stream=True,
                extra_body={
                    "chat_template_kwargs": {"thinking": thinking},
                },
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    reasoning_buffer += rc
                    reasoning_box.markdown(
                        f"🧠 *Reasoning…*\n\n> {reasoning_buffer[-800:]}"
                    )

                if delta and delta.content:
                    full_response += delta.content
                    placeholder.markdown(full_response + "▌")

            reasoning_box.empty()
            placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"API error: {e}")
            full_response = f"⚠️ Error: {e}"

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
