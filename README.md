# 📺 YouTube Summarizer AI

> Chat with YouTube videos and generate structured study notes using RAG and Groq.

## 🌐 Live Demo

**Streamlit:** https://youtube-summari-ai.streamlit.app/

## ✨ Features

- Works with YouTube videos, Shorts and live replays
- Caption-track and language selection
- Multilingual transcript retrieval
- RAG-based conversational Q&A
- Chat memory
- AI-generated study notes
- Notes translation into selected languages
- PDF and Markdown export
- Environment-based API key configuration
- Streamlit interface

## 🧠 How It Works

~~~text
YouTube URL
    ↓
Transcript / Captions
    ↓
Text Chunking
    ↓
Multilingual Embeddings
    ↓
Vector Retrieval
    ↓
Groq LLM
    ↓
Answer / Study Notes
~~~

## 🧰 Tech Stack

- Python
- Streamlit
- Groq
- RAG
- Sentence Transformers
- YouTube transcript processing
- PDF generation
- Markdown export

## 🚀 Setup

~~~bash
git clone https://github.com/Rohankashyap5526/youtube-summarizer-ai.git
cd youtube-summarizer-ai
pip install -r requirements.txt
~~~

Create `.env`:

~~~env
GROQ_API_KEY=your_groq_api_key
~~~

Run:

~~~bash
streamlit run nav.py
~~~

## ⚠️ Notes

Transcript availability depends on the video's available captions. Cloud hosting can also be affected by YouTube rate limiting.

## 📁 Project Structure

~~~text
nav.py       # Main Streamlit entry point
app.py       # Q&A interface
notes.py     # Study-note generation
utils.py     # Transcript, embedding and LLM helpers
assets/      # Fonts and supporting assets
~~~

## 🔐 Security

Never commit `.env` or your Groq API key.

## 📄 License

MIT
