# ============================================================
# IMPORTS
# ============================================================

import re
import io
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from functools import lru_cache

import streamlit as st
import wikipediaapi
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pdfplumber
from docx import Document

# OCR & Image Processing
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


# ============================================================
# CHECKSUM VALIDATION (GÜVENİRLİK İÇİN)
# ============================================================

def luhn_check(card_number: str) -> bool:
    """Kredi kartı numarası geçerliliğini kontrol eder."""
    digits = [int(d) for d in re.sub(r'\D', '', card_number)]
    if not digits: return False
    checksum = digits[-1]
    payload = digits[:-1][::-1]
    total = 0
    for i, d in enumerate(payload):
        if i % 2 == 0:
            d *= 2
            if d > 9: d -= 9
        total += d
    return (total + checksum) % 10 == 0

def tckn_check(tckn: str) -> bool:
    """TC Kimlik Numarası algoritma doğrulaması."""
    if len(tckn) != 11 or tckn[0] == '0': return False
    digits = [int(d) for d in tckn]
    sum_odd = sum(digits[0:9:2])
    sum_even = sum(digits[1:8:2])
    if (sum_odd * 7 - sum_even) % 10 != digits[9]: return False
    if sum(digits[:10]) % 10 != digits[10]: return False
    return True


# ============================================================
# REGEX PATTERNS (GÜNCELLENDİ)
# ============================================================

REGEX_PATTERNS = {
    "TCKN": r"\b[1-9][0-9]{10}\b",
    "PASSPORT": r"\b[A-Z0-9]{6,9}\b",
    "IBAN": r"\bTR\d{2}\s?(\d{4}\s?){5}\d{2}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
    "EMAIL": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "PHONE": r"(?:\+90|0)?\s?\(?[5]\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}",
    "IP": r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
    "DATE": r"\b\d{1,2}[./\s-]?\d{1,2}[./\s-]?\d{2,4}\b"
}

PUBLIC_EMAIL_DOMAINS = ["info@", "support@", "contact@", "@google.com", "@amazon.com"]

def is_public_email(email: str) -> bool:
    email_lower = email.lower()
    return any(email_lower.startswith(p) or email_lower.endswith(p) for p in PUBLIC_EMAIL_DOMAINS)


# ============================================================
# FONT (UNICODE SUPPORT) - TÜRKÇE KARAKTER SORUNU İÇİN
# ============================================================

def register_fonts():
    import urllib.request
    font_path = "Roboto-Regular.ttf"
    
    # 1. Klasörde font yoksa internetten Türkçe destekli Roboto fontunu otomatik indir
    if not os.path.exists(font_path):
        try:
            url = "https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Regular.ttf"
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            pass
            
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont("Roboto", font_path))
        return "Roboto"

    # 2. İndirme başarısız olursa yerel sistemlerdeki (Windows/Mac/Linux) Türkçe fontları ara
    local_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # Linux
        "C:\\Windows\\Fonts\\arial.ttf",                   # Windows
        "/Library/Fonts/Arial.ttf"                         # Mac
    ]
    for path in local_fonts:
        if os.path.exists(path):
            pdfmetrics.registerFont(TTFont("LocalUTF8", path))
            return "LocalUTF8"
            
    return "Helvetica" # Son çare

DEFAULT_FONT = register_fonts()


# ============================================================
# FILE READERS (HYBRID OCR ENTEGRASYONU)
# ============================================================

def read_pdf_smart(file_bytes: bytes) -> str:
    """Önce dijital metni okur, metin yoksa OCR (Tesseract) çalıştırır."""
    text = ""
    # 1. Standart PDF okuma
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t: text += t + "\n"
    
    # 2. Eğer sayfa sayısı var ama metin yoksa (Taranmış PDF)
    if len(text.strip()) < 10:
        with st.status("OCR çalıştırılıyor (Taranmış belge tespit edildi)..."):
            images = convert_from_path(io.BytesIO(file_bytes), dpi=300)
            ocr_text = ""
            for img in images:
                # Türkçe ve İngilizce desteği ile okuma
                ocr_text += pytesseract.image_to_string(img, lang='tur+eng') + "\n"
            return ocr_text
    return text

def read_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Annotation:
    type: str
    value: str
    start: int
    end: int
    source: str
    level: int = None
    action: str = None
    reason: List[str] = field(default_factory=list)
    placeholder: str = None


# ============================================================
# BERT NER & WIKIPEDIA
# ============================================================

class BERTNER:
    MODELS = {
        "en": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "tr": "savasy/bert-base-turkish-ner-cased"
    }
    _pipes = {}

    def __init__(self):
        for lang, model in self.MODELS.items():
            if lang not in self._pipes:
                self._pipes[lang] = pipeline(
                    "ner", model=model, tokenizer=model, aggregation_strategy="max"
                )

    def predict(self, text: str, lang: str):
        return [
            Annotation(type=e["entity_group"], value=e["word"], 
                       start=e["start"], end=e["end"], source="bert")
            for e in self._pipes[lang](text)
        ]

bert_ner = BERTNER()

wiki_en = wikipediaapi.Wikipedia(user_agent="privacy_tool", language="en")
wiki_tr = wikipediaapi.Wikipedia(user_agent="privacy_tool", language="tr")

@lru_cache(maxsize=1000)
def is_public_entity(name, lang):
    wiki = wiki_tr if lang == "tr" else wiki_en
    try:
        page = wiki.page(name)
        if not page.exists() or len(page.text) < 500: return False
        categories = " ".join(page.categories.keys()).lower()
        signals = ["president", "ceo", "politician", "siyasetçi", "şirket", "actor", "company", "founder"]
        return any(s in categories for s in signals)
    except: return False


# ============================================================
# CORE LOGIC: MERGE, SCORE & SMART MASK
# ============================================================

def find_regex_entities(text: str):
    entities = []
    for label, pattern in REGEX_PATTERNS.items():
        for m in re.finditer(pattern, text):
            val = m.group()
            # Checksum Doğrulamaları
            if label == "TCKN" and not tckn_check(val): continue
            if label == "CREDIT_CARD" and not luhn_check(val): continue
            
            entities.append(Annotation(label, val, m.start(), m.end(), "regex"))
    return entities

def merge_entities(regex_entities, bert_entities):
    all_entities = sorted(regex_entities + bert_entities, key=lambda x: (x.start, -(x.end - x.start)))
    merged = []
    priority = {"regex": 2, "bert": 1}
    for e in all_entities:
        if not merged:
            merged.append(e)
            continue
        last = merged[-1]
        if e.start >= last.end:
            merged.append(e)
        elif priority[e.source] > priority[last.source]:
            merged[-1] = e
    return merged

def contextual_scoring(entity, lang):
    etype = entity.type.upper()
    val = entity.value
    if etype in ("PERSON", "ORG") and is_public_entity(val, lang):
        return 3, "PASS", ["public_entity"]
    
    # Duyarlı türler
    if etype in ("TCKN", "CREDIT_CARD", "IBAN", "PASSPORT"):
        return 1, "MASK", ["high_risk_pii"]
    if etype in ("PERSON", "PHONE", "EMAIL", "DATE", "IP"):
        return 2, "MASK", ["personal_data"]
    
    return 3, "PASS", ["low_risk"]

def apply_smart_masking(text, annotations):
    """Pseudonymization: 'Ahmet' -> [PERSON_1]"""
    counters = defaultdict(int)
    mapping = {}
    
    # 1. Mapping oluştur (Aynı değere aynı placeholder)
    for a in annotations:
        if a.action != "MASK": continue
        key = (a.type, a.value.strip().lower())
        if key not in mapping:
            counters[a.type] += 1
            mapping[key] = f"[{a.type}_{counters[a.type]}]"
        a.placeholder = mapping[key]

    # 2. Metni güncelle (Sondan başa)
    sorted_anns = sorted([a for a in annotations if a.placeholder], key=lambda x: x.start, reverse=True)
    for a in sorted_anns:
        text = text[:a.start] + a.placeholder + text[a.end:]
    return text


# ============================================================
# SUMMARY
# ============================================================

def summary_table(annotations):
    lines = ["Value | Type | Level | Action | Reason"]
    for a in annotations:
        lines.append(f"{a.value} | {a.type} | {a.level} | {a.action} | {', '.join(a.reason)}")
    return "\n".join(lines)


# ============================================================
# PDF / DOCX EXPORT
# ============================================================

def export_masked_pdf(text):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    styles = getSampleStyleSheet()
    style = styles["Normal"]
    style.fontName = DEFAULT_FONT
    style.fontSize = 11

    elements = []
    for line in text.split("\n"):
        elements.append(Paragraph(line, style))
        elements.append(Spacer(1, 6))

    doc = SimpleDocTemplate(temp_file.name, pagesize=A4)
    doc.build(elements)
    return temp_file.name

def export_masked_docx(text):
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    doc.save(temp.name)
    return temp.name


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_text(text, lang):
    regex_entities = find_regex_entities(text)
    bert_entities = bert_ner.predict(text, lang)
    merged = merge_entities(regex_entities, bert_entities)

    for e in merged:
        level, action, reason = contextual_scoring(e, lang)
        e.level = level
        e.action = action
        e.reason = reason

    masked = apply_smart_masking(text, merged)
    return summary_table(merged), masked


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="PII Masking Tool", layout="wide")

st.title("PII Masking & NER Tool")

uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

lang = st.selectbox("Language", ["en", "tr"])

if uploaded_file:
    file_bytes = uploaded_file.read()

    if uploaded_file.type == "application/pdf":
        raw_text = read_pdf_smart(file_bytes) # Akıllı okuyucu eklendi (OCR)
    else:
        raw_text = read_docx(file_bytes)

    st.subheader("Original Text")
    st.text_area("", raw_text, height=250)

    summary, masked_text = analyze_text(raw_text, lang)

    st.subheader("Masked Text")
    st.text_area("", masked_text, height=250)

    # İNDİRME SEÇENEKLERİ (DROPDOWN EKLENDİ)
    st.subheader("Download / İndir")
    export_format = st.selectbox("Format Seçin", ["PDF", "DOCX"])

    if export_format == "PDF":
        pdf_path = export_masked_pdf(masked_text)
        st.download_button(
            label="Download Masked PDF",
            data=open(pdf_path, "rb").read(),
            file_name="masked_output.pdf"
        )
    elif export_format == "DOCX":
        docx_path = export_masked_docx(masked_text)
        st.download_button(
            label="Download Masked DOCX",
            data=open(docx_path, "rb").read(),
            file_name="masked_output.docx"
        )
