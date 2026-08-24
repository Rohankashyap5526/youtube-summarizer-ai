import dotenv
import streamlit as st
from utils import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    extract_video_id,
    get_video_metadata,
    is_blocked_error,
    is_proxy_configured,
    list_available_transcripts,
    load_groq_api_key,
)

dotenv.load_dotenv()

st.set_page_config(page_title="🤖 YouTube Summarizer AI", layout="wide")

groq_api = load_groq_api_key()
st.session_state.groq_api = groq_api

pg1 = st.Page("app.py", title="💬 QnA", icon="💬")
pg2 = st.Page("notes.py", title="📝 Note Generation", icon="📝")

with st.sidebar:
    st.header("⚙️ Settings")

    if groq_api:
        st.success("Groq API key loaded from .env", icon="✅")
    else:
        st.error(
            "No Groq API key found.\n\n"
            "Add it to a `.env` file in the project root:\n\n"
            "```\nGROQ_API_KEY=your_key_here\n```",
            icon="🚫",
        )

    if not is_proxy_configured():
        st.caption(
            "ℹ️ No proxy configured. If transcript fetching gets blocked by YouTube "
            "(common on cloud hosts), add `WEBSHARE_PROXY_USERNAME` / "
            "`WEBSHARE_PROXY_PASSWORD` to your `.env`."
        )

    st.session_state.model = st.selectbox(
        "🧠 Model",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda m: AVAILABLE_MODELS[m],
        index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL),
    )

    st.divider()

    new_url = st.text_input(
        "🔗 Paste any YouTube video URL",
        value=st.session_state.get("url", ""),
        placeholder="https://www.youtube.com/watch?v=... (also works with Shorts, youtu.be, live)",
    )

    if new_url != st.session_state.get("url", ""):
        # New video: clear everything tied to the previous one
        for key in ("retriever", "author_name", "title", "notes", "video_id",
                    "available_transcripts", "language_code"):
            st.session_state.pop(key, None)
        st.session_state.chat_messages = []

    st.session_state.url = new_url
    video_id = extract_video_id(new_url) if new_url else None
    st.session_state.video_id = video_id

    if new_url and not video_id:
        st.warning("That doesn't look like a valid YouTube URL.")

    if video_id and groq_api:
        meta = get_video_metadata(video_id)
        st.session_state.title = meta["title"]
        st.session_state.author_name = meta["author"]

        if meta["thumbnail"]:
            st.image(meta["thumbnail"], use_container_width=True)
        st.caption(f"**{meta['title']}**\n\nby {meta['author']}")

        try:
            transcripts = list_available_transcripts(video_id)
            st.session_state.available_transcripts = transcripts
        except Exception as e:
            transcripts = []
            st.session_state.available_transcripts = []
            if is_blocked_error(e):
                st.error(
                    "YouTube is blocking transcript requests from this server's IP "
                    "(common on cloud hosts like Streamlit Cloud, AWS, GCP). Add "
                    "`WEBSHARE_PROXY_USERNAME` / `WEBSHARE_PROXY_PASSWORD` to your "
                    "`.env` — see README.",
                    icon="🚫",
                )
            else:
                st.error(f"Couldn't list transcripts: {e}")

        if transcripts:
            labels = [t["label"] for t in transcripts]
            # Prefer a manually-created English transcript if present, else the first entry
            default_idx = 0
            for i, t in enumerate(transcripts):
                if t["language_code"].startswith("en") and not t["is_generated"]:
                    default_idx = i
                    break
            chosen_label = st.selectbox("🌐 Transcript language", labels, index=default_idx)
            chosen = next(t for t in transcripts if t["label"] == chosen_label)
            st.session_state.language_code = chosen["language_code"]
            st.caption(f"{len(transcripts)} language(s) available for this video.")
        elif video_id:
            st.info("No caption languages found for this video yet.")

pg = st.navigation([pg1, pg2])
pg.run()
