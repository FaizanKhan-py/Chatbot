import os

import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
from PIL import Image, ImageDraw

load_dotenv()

SYSTEM_PROMPT = (
    "You are Faizan's Book Bot, a friendly chatbot that only talks about books. "
    "If anyone asks who you are, what your name is, or who made you, "
    "reply that you are Faizan's Book Bot, a chatbot that tells people about books. "
    "You can help with book summaries, plots, characters, themes, authors, genres, "
    "book recommendations, reading lists, comparisons between books, and literary "
    "context. Be warm, clear, and concise. Avoid spoilers unless the user asks for "
    "them, and say so if you are about to reveal one. If you are not sure a book "
    "exists or you don't know the details, say so instead of making things up. "
    "If the user asks about something unrelated to books, politely say you can only "
    "help with books and invite them to ask a book-related question."
)
TEMPERATURE = 0.7


# ---------- Favicon (drawn in memory, no image file needed) ----------
def make_favicon():
    S = 1024
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, S, S), radius=int(S * 0.22), fill="#111827")
    d.rounded_rectangle((int(S * .20), int(S * .24), int(S * .80), int(S * .66)), radius=int(S * .12), fill="white")
    d.polygon([(int(S * .30), int(S * .62)), (int(S * .30), int(S * .82)), (int(S * .52), int(S * .62))], fill="white")
    r = int(S * .04)
    for x in (.36, .50, .64):
        cx, cy = int(S * x), int(S * .45)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#111827")
    return img.resize((128, 128), Image.LANCZOS)


st.set_page_config(page_title="Faizan's Book Bot", page_icon=make_favicon(), layout="centered")

# ---------- Styling ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, .stApp, [class*="st-"] { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
.stApp { background: #f4f4f5; }

/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* The whole page becomes one card */
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 760px;
    margin: 2.5rem auto;
    padding: 1.4rem 1.6rem !important;
    background: #ffffff;
    border: 1px solid #e7e7ea;
    border-radius: 18px;
    box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 12px 32px rgba(16,24,40,.06);
}

/* Header */
.hdr-title { font-size: 1.05rem; font-weight: 600; color: #111827; line-height: 1.3; }
.hdr-sub { font-size: .8rem; color: #6b7280; margin-top: 2px; }
hr { margin: .9rem 0 .5rem 0 !important; border-color: #eeeef0 !important; }

/* Message bubbles */
[class*="st-key-user-"], [class*="st-key-bot-"] {
    width: fit-content;
    max-width: 85%;
    padding: .7rem 1rem;
    margin-bottom: .6rem;
    border-radius: 16px;
    font-size: .95rem;
    line-height: 1.6;
}
[class*="st-key-user-"] {
    margin-left: auto;
    background: #111827;
    border-bottom-right-radius: 5px;
}
[class*="st-key-user-"] * { color: #ffffff !important; }
[class*="st-key-bot-"] {
    background: #f3f4f6;
    border-bottom-left-radius: 5px;
}
[class*="st-key-bot-"] * { color: #111827; }
[class*="st-key-user-"] p, [class*="st-key-bot-"] p { margin: 0 0 .5rem 0; }
[class*="st-key-user-"] p:last-child, [class*="st-key-bot-"] p:last-child { margin-bottom: 0; }

/* Empty state */
.empty { text-align: center; color: #9ca3af; font-size: .95rem; padding-top: 11rem; }

/* Input */
[data-testid="stChatInput"] { border: 1px solid #e5e7eb; border-radius: 12px; background: #fff; }
[data-testid="stChatInput"]:focus-within { border-color: #111827; box-shadow: 0 0 0 3px rgba(17,24,39,.08); }
[data-testid="stChatInput"] textarea { font-size: .95rem; }

@media (max-width: 640px) {
    .block-container, [data-testid="stMainBlockContainer"] { margin: 0; border-radius: 0; border: none; min-height: 100vh; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Config (from .env, environment, or .streamlit/secrets.toml) ----------
def get_setting(name, default=""):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


ENDPOINT = get_setting("AZURE_OPENAI_ENDPOINT")
API_KEY = get_setting("AZURE_OPENAI_API_KEY")
DEPLOYMENT = get_setting("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
API_VERSION = get_setting("AZURE_OPENAI_API_VERSION", "2024-10-21")
configured = bool(ENDPOINT and API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Header ----------
st.markdown(
    '<div class="hdr-title">📚 Faizan\'s Book Bot</div><div class="hdr-sub">Your guide to books, authors and great reads</div>',
    unsafe_allow_html=True,
)
st.divider()

# ---------- Messages + input, inside the card ----------
messages_box = st.container(height=470, border=False)
input_box = st.container()

with input_box:
    prompt = st.chat_input(
        "Ask me about a book, author or genre" if configured else "Missing Azure endpoint or key in .env",
        disabled=not configured,
    )

with messages_box:
    if not st.session_state.messages and not prompt:
        text = "Which book would you like to talk about today?" if configured else "Add AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY to your .env file."
        st.markdown(f'<div class="empty">{text}</div>', unsafe_allow_html=True)

    for i, m in enumerate(st.session_state.messages):
        kind = "user" if m["role"] == "user" else "bot"
        with st.container(key=f"{kind}-{i}"):
            st.markdown(m["content"])

    if prompt and configured:
        n = len(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.container(key=f"user-{n}"):
            st.markdown(prompt)

        client = AzureOpenAI(azure_endpoint=ENDPOINT, api_key=API_KEY, api_version=API_VERSION)

        with st.container(key=f"bot-{n + 1}"):
            try:
                stream = client.chat.completions.create(
                    model=DEPLOYMENT,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages,
                    temperature=TEMPERATURE,
                    stream=True,
                )

                def token_generator():
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                reply = st.write_stream(token_generator())
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Something went wrong: {e}")