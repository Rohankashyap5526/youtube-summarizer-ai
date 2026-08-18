import warnings

import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from utils import fetch_transcript_text, get_embeddings, get_llm

warnings.filterwarnings("ignore")

groq_api = st.session_state.get("groq_api")
url = st.session_state.get("url")
video_id = st.session_state.get("video_id")
language_code = st.session_state.get("language_code")
model = st.session_state.get("model")


@st.cache_resource(show_spinner=False)
def build_retriever(video_id: str, language_code: str):
    text, language_name = fetch_transcript_text(video_id, language_code)
    doc = Document(page_content=text, metadata={"video_id": video_id, "language": language_name})

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents([doc])

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 5}), language_name


st.title("🤖 YouTube RAG System")
st.subheader("💬 Interactive Q&A")
st.markdown("Ask questions about any YouTube video's transcript — works in every language YouTube provides captions for.")

msgs = StreamlitChatMessageHistory(key="chat_messages")

if not groq_api:
    st.error(
        "No Groq API key found. Add `GROQ_API_KEY=your_key_here` to a `.env` file "
        "in the project root and restart the app."
    )
    st.stop()

if not url:
    st.info("👈 Paste a YouTube URL in the sidebar to get started.")
    st.chat_input("Paste a video URL first...", disabled=True)
    st.stop()

if not video_id:
    st.warning("That doesn't look like a valid YouTube URL. Please check the link in the sidebar.")
    st.stop()

if not language_code:
    st.warning("No transcript language could be determined for this video yet.")
    st.stop()

analyze_clicked = st.button("🔍 Analyze Video", type="primary")

llm = get_llm(groq_api, model=model)

if analyze_clicked:
    with st.spinner("Fetching transcript and building knowledge base..."):
        try:
            build_retriever.clear()  # always rebuild for a fresh click
            retriever, language_name = build_retriever(video_id, language_code)
            st.session_state.retriever = retriever
            st.session_state.transcript_language = language_name
            msgs.clear()
            st.success(f"Analysis complete! Transcript language: {language_name}")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

chat_container = st.container()
with chat_container:
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

if "retriever" not in st.session_state:
    st.info("Click **Analyze Video** above to build the Q&A knowledge base for this video.")
    st.chat_input("Analyze a video first...", disabled=True)
else:
    if query := st.chat_input("Ask a question about the video content (any language):"):
        msgs.add_user_message(query)

        with chat_container:
            st.chat_message("human").write(query)

        with st.spinner("Thinking..."):
            try:
                retrieved_docs = st.session_state.retriever.invoke(query)
                context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

                current_author = st.session_state.get("author_name", "the speaker")
                transcript_language = st.session_state.get("transcript_language", "the video's original language")

                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            f"""You are assisting with a YouTube video by {current_author}. \
The video's transcript is in {transcript_language}. \
Answer the user's question using only the context below. \
Always reply in the same language the user asked the question in. \
Structure your response with headings and bullet points where helpful. \
If the answer isn't in the context, say so honestly instead of guessing.

Context:
{context_text}""",
                        ),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{question}"),
                    ]
                )

                chain = prompt | llm

                chain_with_history = RunnableWithMessageHistory(
                    chain,
                    lambda session_id: msgs,
                    input_messages_key="question",
                    history_messages_key="history",
                )

                config = {"configurable": {"session_id": f"youtube_{video_id}"}}
                response = chain_with_history.invoke({"question": query}, config)

                with chat_container:
                    st.chat_message("ai").write(response.content)

            except Exception as e:
                st.error(f"An error occurred while generating the answer: {e}")
