import streamlit as st
import whisper
import tempfile
import os
from transformers import pipeline
from fpdf import FPDF
from gtts import gTTS
import io
from fpdf import FPDF
from fpdf import FPDF
import tempfile
from pathlib import Path

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI-Powered Speech-to-Text & Summarization", layout="wide")

# -------------------- TITLE --------------------
st.title("🎙️ AI-Powered Speech-to-Text and Summarization System")
st.markdown("Upload your meeting or lecture audio, and let AI transcribe & summarize it automatically.")

# -------------------- FILE UPLOAD --------------------
uploaded_audio = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a"])

# -------------------- SETTINGS --------------------
col1, col2 = st.columns(2)
with col1:
    whisper_model = st.selectbox("Select Speech Recognition Model", ["tiny", "base", "small", "medium", "large"], index=2)
with col2:
    summarizer_model = st.selectbox("Select Summarization Model", ["t5-small", "t5-base", "google/pegasus-xsum"], index=1)

# -------------------- FUNCTIONS --------------------
@st.cache_resource
def load_whisper_model(name):
    return whisper.load_model(name)

@st.cache_resource
def load_summarizer(name):
    return pipeline("summarization", model=name)

def transcribe_audio(path, model_name):
    model = load_whisper_model(model_name)
    result = model.transcribe(path)
    return result["text"]

def summarize_text(text, model_name):
    summarizer = load_summarizer(model_name)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summary = ""
    for chunk in chunks:
        out = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summary += out[0]['summary_text'] + " "
    return summary.strip()

def generate_pdf(transcript, summary):
    font_path = Path("DejavuSans.ttf")  

    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", str(font_path), uni=True)
    pdf.set_font("DejaVu", size=12)

    pdf.multi_cell(0, 10, txt="🎙️ AI-Powered Speech-to-Text and Summarization Report")
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt="🗣️ Transcript:\n" + transcript)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt="📝 Summary:\n" + summary)

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name

# -------------------- MAIN LOGIC --------------------
if uploaded_audio is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_audio.getbuffer())
        tmp_path = tmp_file.name

    if st.button("Start Transcription and Summarization ▶️"):
        with st.spinner("Transcribing audio... ⏳"):
            transcript = transcribe_audio(tmp_path, whisper_model)
        st.success("✅ Transcription completed!")
        st.text_area("Transcript", transcript, height=200)

        with st.spinner("Summarizing transcript... 🧠"):
            summary = summarize_text(transcript, summarizer_model)
        st.success("✅ Summarization completed!")
        st.text_area("Summary", summary, height=150)

        # Generate and download summary speech
        with st.spinner("Generating summary speech (TTS)... 🔊"):
            tts = gTTS(text=summary, lang='en')
            mp3_bytes = io.BytesIO()
            tts.write_to_fp(mp3_bytes)
            mp3_bytes.seek(0)
        st.audio(mp3_bytes, format="audio/mp3")

        # PDF Download
        pdf_path = generate_pdf(transcript, summary)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Report (PDF)", f, file_name="Meeting_Summary.pdf")

        # Text download
        st.download_button("🗒️ Download Summary (Text)", summary, file_name="summary.txt")
else:
    st.info("👆 Upload an audio file to begin.")
