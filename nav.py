"""
Shared helpers for the YouTube Summarizer AI app.

Handles:
- Extracting a video ID from any YouTube URL shape (watch, youtu.be, shorts, live, embed, mobile)
- Fetching video title / author via YouTube's public oEmbed endpoint (no API key needed)
- Listing every available transcript (manual + auto-generated) via yt-dlp
- Fetching transcript text as clean plain text (via yt-dlp, parsed from VTT)
- Building the Groq LLM client and a multilingual embedding model

NOTE: Transcript fetching uses yt-dlp instead of youtube_transcript_api. yt-dlp hits
different YouTube endpoints and, in practice, gets IP-blocked far less often on cloud
hosts. Requires: pip install yt-dlp
"""

import os
import re
from functools import lru_cache

import requests
import streamlit as st
import yt_dlp
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# Groq models — llama-3.3-70b-versatile / llama-3.1-8b-instant are deprecated
# by Groq (shutdown 2026-08-16). We default to their recommended replacements.
# ---------------------------------------------------------------------------
AVAILABLE_MODELS = {
    "openai/gpt-oss-120b": "GPT-OSS 120B — best quality (recommended)",
    "openai/gpt-oss-20b": "GPT-OSS 20B — fastest / cheapest",
}
DEFAULT_MODEL = "openai/gpt-oss-120b"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_VIDEO_ID_PATTERNS = [
    r"(?:v=|/videos/|embed/|shorts/|live/|v/)([0-9A-Za-z_-]{11})",
    r"youtu\.be/([0-9A-Za-z_-]{11})",
]


def extract_video_id(url: str) -> str | None:
    """Extract the 11-char YouTube video ID from any valid YouTube URL shape."""
    if not url:
        return None
    url = url.strip()
    for pattern in _VIDEO_ID_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Bare video ID pasted directly
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url):
        return url
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def get_video_metadata(video_id: str) -> dict:
    """Fetch title/author/thumbnail via YouTube's public oEmbed endpoint (no API key required)."""
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "title": data.get("title", "Untitled Video"),
            "author": data.get("author_name", "Unknown Creator"),
            "thumbnail": data.get("thumbnail_url"),
        }
    except Exception:
        return {"title": "Untitled Video", "author": "Unknown Creator", "thumbnail": None}


# ---------------------------------------------------------------------------
# yt-dlp based transcript fetching
# ---------------------------------------------------------------------------

class TranscriptError(Exception):
    """Human-readable transcript failure."""


def _ytdlp_opts() -> dict:
    opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "vtt",
    }
    # Optional proxy support (still useful if yt-dlp itself gets rate-limited)
    proxy_url = os.getenv("YTDLP_PROXY_URL") or os.getenv("HTTPS_PROXY_URL")
    if proxy_url:
        opts["proxy"] = proxy_url
    return opts


@lru_cache(maxsize=64)
def _ytdlp_extract_info(video_id: str) -> dict:
    """
    Run yt-dlp's metadata extraction once per video_id (cached — this is the
    network call that replaces youtube_transcript_api.list()).
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(_ytdlp_opts()) as ydl:
            return ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        msg = str(e).lower()
        if "private" in msg:
            raise TranscriptError("This video is private.")
        if "unavailable" in msg or "removed" in msg:
            raise TranscriptError("This video is unavailable. Double-check the URL.")
        if "age" in msg and "restrict" in msg:
            raise TranscriptError("This video is age-restricted and its transcript can't be accessed.")
        if "429" in msg or "too many requests" in msg:
            raise TranscriptError(
                "YouTube rate-limited this server. Wait a bit and try again, or set "
                "YTDLP_PROXY_URL in your .env to route through a proxy."
            )
        raise TranscriptError(f"yt-dlp couldn't process this video: {e}")


_LANG_NAMES = {
    "en": "English", "en-US": "English (US)", "en-GB": "English (UK)",
    "es": "Spanish", "fr": "French", "de": "German", "hi": "Hindi",
    "pt": "Portuguese", "ru": "Russian", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "ar": "Arabic", "it": "Italian",
}


def _lang_label(code: str) -> str:
    return _LANG_NAMES.get(code, code)


@st.cache_data(show_spinner=False, ttl=3600)
def list_available_transcripts(video_id: str) -> list[dict]:
    """
    Return every transcript available for this video (manual + auto-generated),
    as a list of picklable dicts: {label, language_code, language, is_generated}
    """
    try:
        info = _ytdlp_extract_info(video_id)
    except TranscriptError as e:
        raise Exception(str(e))

    entries = []
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}

    for code in manual:
        entries.append({
            "label": f"{_lang_label(code)} ({code}) · manual",
            "language_code": code,
            "language": _lang_label(code),
            "is_generated": False,
        })
    for code in auto:
        if code in manual:
            continue  # prefer the manual entry, don't list both
        entries.append({
            "label": f"{_lang_label(code)} ({code}) · auto-generated",
            "language_code": code,
            "language": _lang_label(code),
            "is_generated": True,
        })

    entries.sort(key=lambda e: (e["is_generated"], e["language"]))
    return entries


def _parse_vtt(vtt_text: str) -> str:
    """Turn raw WebVTT into deduped plain text (auto-captions repeat rolling lines)."""
    lines = []
    for raw in vtt_text.splitlines():
        line = raw.strip()
        if not line or line == "WEBVTT":
            continue
        if "-->" in line:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        # strip inline VTT tags like <00:00:01.000><c> word</c>
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line:
            lines.append(line)

    # Dedupe consecutive duplicate/overlapping lines (common in rolling auto-captions)
    deduped = []
    for line in lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)

    text = " ".join(deduped)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_transcript_text(video_id: str, language_code: str) -> tuple[str, str]:
    """
    Fetch a transcript in the given language and return (plain_text, language_name).
    Raises a human-readable Exception on failure.
    """
    try:
        info = _ytdlp_extract_info(video_id)
    except TranscriptError as e:
        raise Exception(str(e))

    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}

    track = None
    is_generated = False
    if language_code in manual:
        track = manual[language_code]
    elif language_code in auto:
        track = auto[language_code]
        is_generated = True

    if not track:
        raise Exception("No transcript is available in the selected language for this video.")

    # Pick the vtt format entry (fall back to the first available if vtt missing)
    fmt = next((f for f in track if f.get("ext") == "vtt"), track[0])
    caption_url = fmt.get("url")
    if not caption_url:
        raise Exception("Could not resolve the caption file URL for this video.")

    try:
        resp = requests.get(caption_url, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise Exception(
            f"Network error downloading captions ({'auto-generated' if is_generated else 'manual'}): {e}"
        )

    text = _parse_vtt(resp.text)
    if not text:
        raise Exception("The transcript for this video is empty.")

    return text, _lang_label(language_code)


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str, model: str = DEFAULT_MODEL, temperature: float = 0.3):
    return ChatGroq(api_key=api_key, model=model, temperature=temperature)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_groq_api_key() -> str | None:
    """API key is read exclusively from the environment (.env). No sidebar input."""
    return os.getenv("GROQ_API_KEY")
