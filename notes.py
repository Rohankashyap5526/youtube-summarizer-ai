import os
import re
import warnings

import streamlit as st
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import fetch_transcript_text, get_llm

warnings.filterwarnings("ignore")

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets", "fonts")
FONT_REGULAR = os.path.join(ASSETS_DIR, "DejaVuSans.ttf")
FONT_BOLD = os.path.join(ASSETS_DIR, "DejaVuSans-Bold.ttf")

groq_api = st.session_state.get("groq_api")
url = st.session_state.get("url")
video_id = st.session_state.get("video_id")
language_code = st.session_state.get("language_code")
model = st.session_state.get("model")

OUTPUT_LANGUAGES = {
    "auto": "Same as video (no translation)",
    "English": "English",
    "Hindi": "Hindi (हिन्दी)",
    "Spanish": "Spanish (Español)",
    "French": "French (Français)",
    "German": "German (Deutsch)",
    "Portuguese": "Portuguese (Português)",
    "Arabic": "Arabic (العربية)",
    "Chinese (Simplified)": "Chinese Simplified (简体中文)",
    "Japanese": "Japanese (日本語)",
    "Korean": "Korean (한국어)",
    "Russian": "Russian (Русский)",
    "Italian": "Italian (Italiano)",
    "Turkish": "Turkish (Türkçe)",
    "Bengali": "Bengali (বাংলা)",
    "Indonesian": "Indonesian (Bahasa Indonesia)",
    "Vietnamese": "Vietnamese (Tiếng Việt)",
}


class UnicodePDF(FPDF):
    """A4 PDF that renders any Unicode script (Latin, Cyrillic, Greek, Vietnamese,
    and most accented/extended Latin) via a bundled DejaVu Sans font."""

    def __init__(self):
        super().__init__()
        self.is_first_page = True
        self.add_font("DejaVu", "", FONT_REGULAR)
        self.add_font("DejaVu", "B", FONT_BOLD)
        self.add_font("DejaVu", "I", FONT_REGULAR)

    def header(self):
        if not self.is_first_page:
            self.set_text_color(140, 140, 140)
            self.set_font("DejaVu", "I", 8)
            self.cell(0, 10, "Study Guide & Transcript Analysis", align="R")
            self.set_y(self.get_y() + 10)
            self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    @staticmethod
    def clean_text(text):
        text = text.replace("**", "")
        return re.sub(r"^[#*\-\s]+", "", text).strip()

    def render_document(self, title, text):
        self.set_font("DejaVu", "B", 22)
        self.set_text_color(20, 30, 45)
        self.multi_cell(0, 12, self.clean_text(title), align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(70, 130, 180)
        self.line(15, self.get_y() + 4, 195, self.get_y() + 4)
        self.ln(12)
        self.is_first_page = False

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                self.ln(4)
                continue

            is_h1 = line.startswith("# ")
            is_h2 = line.startswith("## ")
            is_h3 = line.startswith("### ")
            is_bullet = line.startswith("*") or line.startswith("-")
            clean_line = self.clean_text(line)
            if not clean_line:
                continue

            if is_h1:
                self.set_font("DejaVu", "B", 17)
                self.set_text_color(20, 30, 45)
                self.multi_cell(0, 10, clean_line, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                continue
            elif is_h2:
                self.set_font("DejaVu", "B", 14)
                self.set_text_color(40, 60, 90)
                self.multi_cell(0, 8, clean_line, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                continue
            elif is_h3:
                self.set_font("DejaVu", "B", 12)
                self.set_text_color(70, 130, 180)
                self.multi_cell(0, 7, clean_line, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                continue

            if is_bullet:
                clean_line = f"• {clean_line}"

            self.set_font("DejaVu", "", 10)
            self.set_text_color(20, 20, 20)
            self.multi_cell(0, 6, clean_line, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(2)


def generate_pdf(text, title):
    pdf = UnicodePDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 20, 15)
    pdf.add_page()
    pdf.render_document(title, text)
    return bytes(pdf.output())


def clean_raw_llm_text(text):
    text = re.sub(r"\bH[1-3]\.\d+\.?\s*", "", text)
    text = re.sub(r"\$(.*?)\$", r"\1", text)
    return text


def safe_filename(title, video_id):
    stem = re.sub(r"[^\w\-]+", "_", title, flags=re.UNICODE).strip("_")
    if not stem:
        stem = video_id
    return stem[:40]


st.title("🤖 YouTube RAG System")

col1, col2, col3 = st.columns([2, 1.4, 1.6])
with col1:
    st.subheader("📝 Notes Generator")
with col2:
    output_lang = st.selectbox(
        "Output language",
        options=list(OUTPUT_LANGUAGES.keys()),
        format_func=lambda k: OUTPUT_LANGUAGES[k],
        label_visibility="collapsed",
    )
with col3:
    generate_btn = st.button("GENERATE NOTES", type="primary", use_container_width=True)

st.markdown("Extract insights and generate a downloadable, well-structured study guide from any YouTube video — in any language.")

if not groq_api:
    st.error(
        "No Groq API key found. Add `GROQ_API_KEY=your_key_here` to a `.env` file "
        "in the project root and restart the app."
    )
    st.stop()

if not url or not video_id:
    st.info("👈 Paste a YouTube URL in the sidebar to get started.")
    st.stop()

if not language_code:
    st.warning("No transcript language could be determined for this video yet.")
    st.stop()

llm = get_llm(groq_api, model=model)

if generate_btn:
    with st.spinner("Fetching transcript..."):
        try:
            full_transcript, transcript_language = fetch_transcript_text(video_id, language_code)
        except Exception as e:
            st.error(str(e))
            st.stop()

    video_title = st.session_state.get("title", "Generated Study Guide")
    target_language = transcript_language if output_lang == "auto" else output_lang

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)
    chunks = text_splitter.split_text(full_transcript)

    partial_summaries = []
    progress_bar = st.progress(0, text="Extracting key concepts...")

    try:
        for i, chunk in enumerate(chunks):
            chunk_prompt = f"""Act as a technical scribe. Extract core concepts from this transcript
section ({i + 1}/{len(chunks)}) of '{video_title}' (original language: {transcript_language}).
STRICT RULES:
- Output ONLY dense bullet points, written in {target_language}.
- NO introductory text or conversational filler.
- Focus on technical definitions, workflows, and factual data.
- Use 'concept: explanation' format for brevity.

Transcript:
{chunk}
"""
            response = llm.invoke(chunk_prompt)
            partial_summaries.append(response.content)
            progress_bar.progress((i + 1) / len(chunks), text=f"Extracting key concepts... ({i + 1}/{len(chunks)})")

        combined_notes = "\n\n".join(partial_summaries)

        progress_bar.progress(1.0, text="Polishing final study guide...")
        if len(combined_notes) < 12000:
            final_prompt = f"""Convert these notes into a high-density, professional study guide for
'{video_title}', written entirely in {target_language}.
CRITICAL FORMATTING & CONTENT RULES:
1. NO RAW MARKS: Do not output layout markers like 'H2.1' or bullet dashes next to headers
   (e.g., use 'Executive Summary', NOT '- Executive Summary').
2. NO MATH SYMBOLS ($): Convert all math formulas into clean plain text. Never use '$'.
   For example, rewrite '$Y=f(X)$' to 'Y = f(X)'.
3. NO REDUNDANCY: Do not repeat definitions. Combine repetitive sections into a single, cohesive breakdown.
4. TYPOGRAPHY: Use standard Markdown formatting (# for main title, ## for themes, ### for sub-topics).
5. LANGUAGE: The entire output must be written in {target_language}, regardless of the source transcript's language.

Raw Notes to Refine:
{combined_notes}
"""
            final_response = llm.invoke(final_prompt)
            notes_text = final_response.content
        else:
            notes_text = combined_notes

        progress_bar.empty()
        st.session_state.notes = notes_text
        st.session_state.title = video_title
        st.session_state.notes_language = target_language
        st.success("Notes generated successfully!")

    except Exception as e:
        progress_bar.empty()
        st.error(f"Error occurred while generating notes: {e}")

if "notes" in st.session_state:
    st.divider()
    st.markdown(f"### Notes for: {st.session_state.title}")
    st.caption(f"Language: {st.session_state.get('notes_language', 'auto')}")
    polished_notes = clean_raw_llm_text(st.session_state.notes)
    st.markdown(polished_notes)

    st.divider()
    dcol1, dcol2 = st.columns(2)

    with dcol1:
        try:
            pdf_data = generate_pdf(polished_notes, st.session_state.title)
            st.download_button(
                label="📥 Download as PDF",
                data=pdf_data,
                file_name=f"{safe_filename(st.session_state.title, video_id)}_notes.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"PDF export failed ({e}). Use the Markdown download instead.")

    with dcol2:
        st.download_button(
            label="📄 Download as Markdown (.md)",
            data=polished_notes.encode("utf-8"),
            file_name=f"{safe_filename(st.session_state.title, video_id)}_notes.md",
            mime="text/markdown",
            use_container_width=True,
        )
