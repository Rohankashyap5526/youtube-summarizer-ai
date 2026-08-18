# 🤖 YouTube Summarizer AI

A Streamlit app that lets you **chat with** and **generate study notes from**
any YouTube video, in **any language YouTube provides captions for** — powered
by Groq's ultra-fast LLM inference.

## ✨ Features

- **Works with any YouTube video** — regular videos, Shorts, live replays, `youtu.be` links.
- **Every language** — automatically lists every caption track a video has (manual
  and auto-generated) and lets you pick which one to use. Retrieval uses a
  multilingual embedding model, so Q&A works in the transcript's own language.
- **💬 Interactive Q&A** — Retrieval-Augmented Generation (RAG) over the video transcript with chat memory.
- **📝 Notes Generator** — turns a transcript into a dense, well-structured study guide,
  optionally **translated into a different output language**, downloadable as PDF or Markdown.
- **Unicode PDF export** — bundled DejaVu Sans font renders Latin, Cyrillic, Greek,
  Vietnamese and other extended-Latin scripts correctly (not just ASCII). A Markdown
  download is always available as a universal fallback for every script (e.g. CJK, Arabic, Devanagari).
- **API key stays in `.env`** — there's no key input in the UI; the app reads
  `GROQ_API_KEY` from the environment only.

## 🚀 Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Add your Groq API key**

   Copy `.env.example` to `.env` and fill in your key (get a free one at
   [console.groq.com/keys](https://console.groq.com/keys)):

   ```bash
   cp .env.example .env
   ```

   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Run the app**

   ```bash
   streamlit run nav.py
   ```

## 🧠 Models

The app defaults to `openai/gpt-oss-120b` on Groq (Groq's recommended
replacement for the now-deprecated `llama-3.3-70b-versatile`). You can switch
to the smaller/faster `openai/gpt-oss-20b` from the sidebar.

## 🌐 A note on language support

- **Transcript language** — chosen per-video in the sidebar, from whatever
  caption tracks YouTube actually has for that video.
- **Notes output language** — chosen on the Notes page; the LLM translates/writes
  the study guide in your selected language regardless of the transcript's language.
- **PDF rendering** — the bundled font (DejaVu Sans) covers Latin, Cyrillic, Greek,
  and Vietnamese well. For scripts it doesn't cover (e.g. Chinese, Japanese, Korean,
  Arabic, Hindi/Devanagari), use the **Markdown download** instead — it always
  renders correctly since it's just UTF-8 text.

## ⚠️ Troubleshooting

- **"No transcript available"** — the video owner disabled captions for that video;
  there's nothing the app can do about that.
- **"YouTube is temporarily blocking transcript requests..."** — some cloud hosts'
  IPs get rate-limited by YouTube. Set `HTTP_PROXY_URL` / `HTTPS_PROXY_URL` in `.env`
  to route transcript requests through a proxy.

## 📁 Project structure

```
nav.py       # Entry point: page config, sidebar (URL + language + model), no API key input
app.py       # Q&A page (RAG chat)
notes.py     # Notes/study-guide generator + PDF/Markdown export
utils.py     # Shared helpers: video ID parsing, oEmbed metadata, transcript fetching, LLM/embeddings
assets/fonts # Bundled DejaVu Sans fonts for Unicode PDF export
```
