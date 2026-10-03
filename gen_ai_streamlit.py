import time
import uuid
import streamlit as st
from openai import OpenAI


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pushkar's AI App",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# NVIDIA API KEY
# ============================================================

api_key = st.secrets.get("NVIDIA_API_KEY")

if not api_key:
    st.error(
        "NVIDIA API key not found.\n\n"
        "Please add NVIDIA_API_KEY to "
        ".streamlit/secrets.toml"
    )
    st.stop()


# ============================================================
# NVIDIA CLIENT
# ============================================================

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)


# ============================================================
# MODEL
# ============================================================

MODEL = "deepseek-v4-pro-0813"


# ============================================================
# SESSION STATE
# ============================================================

if "chats" not in st.session_state:

    first_chat_id = str(uuid.uuid4())[:8]

    st.session_state.chats = {
        first_chat_id: {
            "title": "New Chat",
            "messages": []
        }
    }

    st.session_state.current_chat_id = first_chat_id


if st.session_state.current_chat_id not in st.session_state.chats:

    st.session_state.current_chat_id = list(
        st.session_state.chats.keys()
    )[0]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("💬 Chat History")

    # New chat
    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        new_chat_id = str(uuid.uuid4())[:8]

        st.session_state.chats[new_chat_id] = {
            "title": "New Chat",
            "messages": []
        }

        st.session_state.current_chat_id = new_chat_id

        st.rerun()

    st.divider()

    # Chat history
    for chat_id, chat_data in st.session_state.chats.items():

        if chat_id == st.session_state.current_chat_id:
            label = f"📌 {chat_data['title']}"
        else:
            label = f"💬 {chat_data['title']}"

        if st.button(
            label,
            key=f"chat_{chat_id}",
            use_container_width=True
        ):

            st.session_state.current_chat_id = chat_id

            st.rerun()


# ============================================================
# ACTIVE CHAT
# ============================================================

active_chat = st.session_state.chats[
    st.session_state.current_chat_id
]

messages = active_chat["messages"]


# ============================================================
# MAIN UI
# ============================================================

st.title("Welcome to Pushkar's AI App")


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# DEEPSEEK FUNCTION
# ============================================================

def get_deepseek_response(chat_messages):

    # --------------------------------------------------------
    # NVIDIA API messages
    # --------------------------------------------------------

    api_messages = [
        {
            "role": "system",
            "content": (
                "You are Pushkar's AI assistant. "
                "Give clear, helpful and accurate answers. "
                "Use simple language when explaining technical topics."
            )
        }
    ]

    for message in chat_messages:

        api_messages.append(
            {
                "role": message["role"],
                "content": message["content"]
            }
        )


    # --------------------------------------------------------
    # RETRY LOGIC
    # --------------------------------------------------------

    for attempt in range(3):

        try:

            response = client.chat.completions.create(

                model=MODEL,

                messages=api_messages,

                temperature=1.0,

                top_p=0.95,

                max_tokens=4096,

                reasoning_effort="none",

                stream=False
            )


            # ------------------------------------------------
            # GET RESPONSE
            # ------------------------------------------------

            answer = response.choices[0].message.content

            if answer:

                return answer

            return "⚠️ DeepSeek returned an empty response."


        except Exception as e:

            error = str(e)

            # Print actual error in terminal
            print("\n" + "=" * 60)
            print("NVIDIA API ERROR")
            print("=" * 60)
            print(error)
            print("=" * 60 + "\n")


            # ------------------------------------------------
            # TEMPORARY SERVER ERROR
            # ------------------------------------------------

            if any(
                code in error
                for code in ["500", "502", "503", "504"]
            ):

                if attempt < 2:

                    wait_time = 2 ** attempt

                    time.sleep(wait_time)

                    continue

                return (
                    "⚠️ **NVIDIA Server Busy**\n\n"
                    "The DeepSeek server is temporarily "
                    "unavailable. Please try again."
                )


            # ------------------------------------------------
            # RATE LIMIT
            # ------------------------------------------------

            if (
                "429" in error
                or "rate limit" in error.lower()
                or "too many requests" in error.lower()
            ):

                return (
                    "⚠️ **Rate Limit Reached**\n\n"
                    "The NVIDIA API rate limit has been "
                    "reached. Please wait and try again."
                )


            # ------------------------------------------------
            # AUTHENTICATION ERROR
            # ------------------------------------------------

            if (
                "401" in error
                or "unauthorized" in error.lower()
                or "authentication" in error.lower()
            ):

                return (
                    "⚠️ **API Key Error**\n\n"
                    "Please check your NVIDIA API key "
                    "in `.streamlit/secrets.toml`."
                )


            # ------------------------------------------------
            # PAYMENT / AVAILABILITY
            # ------------------------------------------------

            if (
                "402" in error
                or "payment required" in error.lower()
            ):

                return (
                    "⚠️ **Model Availability Error**\n\n"
                    "The NVIDIA Free Endpoint for this model "
                    "may not be available. Check the model "
                    "availability on NVIDIA NIM."
                )


            # ------------------------------------------------
            # OTHER ERROR
            # ------------------------------------------------

            return (
                "⚠️ **NVIDIA API Error**\n\n"
                f"```text\n{error}\n```"
            )


    return "⚠️ DeepSeek is temporarily unavailable."


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input("What's on your mind?")


if prompt:

    # ========================================================
    # CREATE CHAT TITLE
    # ========================================================

    if active_chat["title"] == "New Chat":

        title = prompt.strip()

        if len(title) > 22:

            title = title[:22] + "..."

        active_chat["title"] = title


    # ========================================================
    # DISPLAY USER MESSAGE
    # ========================================================

    with st.chat_message("user"):

        st.markdown(prompt)


    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # ========================================================
    # DEEPSEEK RESPONSE
    # ========================================================

    with st.chat_message("assistant"):

        with st.spinner("DeepSeek is thinking..."):

            response_text = get_deepseek_response(messages)

        st.markdown(response_text)


    # ========================================================
    # SAVE ASSISTANT RESPONSE
    # ========================================================

    messages.append(
        {
            "role": "assistant",
            "content": response_text
        }
    )


    # ========================================================
    # REFRESH
    # ========================================================

    st.rerun()
