"""
Shared helpers for the YouTube Summarizer AI app.

Handles:
- Extracting a video ID from any YouTube URL shape (watch, youtu.be, shorts, live, embed, mobile)
- Fetching video title / author via YouTube's public oEmbed endpoint (no API key needed)
- Listing every available transcript (manual + auto-generated) in every language YouTube offers
- Fetching transcript text as clean plain text (not raw SRT/XML)
- Building the Groq LLM client and a multilingual embedding model
"""

import os
import re
from functools import lru_cache

import requests
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    RequestBlocked,
    IpBlocked,
    AgeRestricted,
    VideoUnplayable,
    PoTokenRequired,
)

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


def _get_proxy_config():
    """Optional proxy support, configured only via .env — helps when a host's
    IP is rate-limited/blocked by YouTube (common on cloud deployments)."""
    http_url = os.getenv("HTTP_PROXY_URL")
    https_url = os.getenv("HTTPS_PROXY_URL")
    if http_url or https_url:
        return GenericProxyConfig(http_url=http_url, https_url=https_url)
    return None


def _transcript_api() -> YouTubeTranscriptApi:
    return YouTubeTranscriptApi(proxy_config=_get_proxy_config())


@st.cache_data(show_spinner=False, ttl=3600)
def list_available_transcripts(video_id: str) -> list[dict]:
    """
    Return every transcript YouTube offers for this video, manual and
    auto-generated, in every language, as a list of picklable dicts:
    {label, language_code, is_generated}
    """
    api = _transcript_api()
    transcript_list = api.list(video_id)

    entries = []
    for t in transcript_list:
        kind = "auto-generated" if t.is_generated else "manual"
        entries.append(
            {
                "label": f"{t.language} ({t.language_code}) · {kind}",
                "language_code": t.language_code,
                "language": t.language,
                "is_generated": t.is_generated,
            }
        )
    # Manually created transcripts first, then alphabetical by language name
    entries.sort(key=lambda e: (e["is_generated"], e["language"]))
    return entries


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_transcript_text(video_id: str, language_code: str) -> tuple[str, str]:
    """
    Fetch a transcript in the given language and return (plain_text, language_name).
    Raises a human-readable Exception on failure.
    """
    api = _transcript_api()
    try:
        transcript_list = api.list(video_id)
        transcript = transcript_list.find_transcript([language_code])
        fetched = transcript.fetch()
        text = " ".join(snippet.text.strip() for snippet in fetched if snippet.text.strip())
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            raise ValueError("The transcript for this video is empty.")
        return text, transcript.language

    except TranscriptsDisabled:
        raise Exception("Captions/transcripts are disabled by the video owner for this video.")
    except NoTranscriptFound:
        raise Exception("No transcript is available in the selected language for this video.")
    except (RequestBlocked, IpBlocked):
        raise Exception(
            "YouTube is temporarily blocking transcript requests from this server's IP. "
            "This is common on cloud-hosted apps. Configure HTTP_PROXY_URL / HTTPS_PROXY_URL "
            "in your .env to work around it, or try again later."
        )
    except AgeRestricted:
        raise Exception("This video is age-restricted and its transcript can't be accessed.")
    except VideoUnplayable:
        raise Exception("This video is unplayable (private, removed, or region-locked).")
    except PoTokenRequired:
        raise Exception("YouTube requires additional verification for this video right now. Please try again later.")
    except VideoUnavailable:
        raise Exception("This video is unavailable. Double-check the URL.")
    except Exception as e:
        raise Exception(f"Could not fetch transcript: {e}")


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str, model: str = DEFAULT_MODEL, temperature: float = 0.3):
    return ChatGroq(api_key=api_key, model=model, temperature=temperature)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_groq_api_key() -> str | None:
    """API key is read exclusively from the environment (.env). No sidebar input."""
    return os.getenv("GROQ_API_KEY")
