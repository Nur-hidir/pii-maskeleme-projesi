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
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

# ============================================================
# REGEX PATTERNS
# ============================================================
REGEX_PATTERNS = {
    "TCKN": r"\b[1-9][0-9]{10}\b",
    "PASSPORT": r"\b[A-Z0-9]{6,9}\b",
    "IBAN": r"\bTR\d{2}\d{4}\d{4}\d{4}\d{4}\d{4}\d{2}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
    "EMAIL": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "PHONE": r"(?:\+90|0)?\s?\(?[5]\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}",
    "IP": r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
    "DATE": r"\b\d{1,2}[./\s-]?\d{1,2}[./\s-]?\d{2,4}\b"
}

PUBLIC_EMAIL_DOMAINS = [
    "info@", "support@", "contact@", "admin@", "help@", "press@", "marketing@", "sales@",
    "@amazon.com", "@spacex.com", "@google.com", "@microsoft.com", "@apple.com",
    "@facebook.com", "@twitter.com", "@linkedin.com", "@hollywood.com"
]

def is_public_email(email: str) -> bool:
    email_lower = email.lower()
    return any(email_lower.startswith(p) or email_lower.endswith(p) for p in PUBLIC_EMAIL_DOMAINS)

# ============================================================
# FONT REGISTRATION FOR PDF
# ============================================================
def register_fonts() -> str:
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    ]
    for path in font_paths:
        if os.path.exists(path):
            pdfmetrics.registerFont(TTFont("CustomUnicodeFont", path))
            return "CustomUnicodeFont"
    return "Helvetica"

DEFAULT_FONT = register_fonts()

# ============================================================
# FILE READING UTILITIES
# ============================================================
def read_pdf(file_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
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
# BERT NER PIPELINE
# ============================================================
class BERTNER:
    MODELS = {
        "en": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "tr": "savasy/bert-base-turkish-ner-cased"
    }
    _pipes: Dict[str, Any] = {}

    def __init__(self):
        for lang, model_name in self.MODELS.items():
            if lang not in self._pipes:
                self._pipes[lang] = pipeline(
                    "ner",
                    model=AutoModelForTokenClassification.from_pretrained(model_name),
                    tokenizer=AutoTokenizer.from_pretrained(model_name),
                    aggregation_strategy="max"
                )

    def predict(self, text: str, lang: str) -> List[Annotation]:
        return [
            Annotation(
                type=e["entity_group"],
                value=e["word"],
                start=e["start"],
                end=e["end"],
                source="bert"
            ) for e in self._pipes[lang](text)
        ]

bert_ner = BERTNER()

# ============================================================
# REGEX ENTITY DETECTION
# ============================================================
def find_regex_entities(text: str) -> List[Annotation]:
    entities = []
    for label, pattern in REGEX_PATTERNS.items():
        for m in re.finditer(pattern, text):
            entities.append(Annotation(label, m.group(), m.start(), m.end(), "regex"))
    return entities

# ============================================================
# MERGE REGEX + BERT ENTITIES
# ============================================================
def merge_entities(regex_entities: List[Annotation], bert_entities: List[Annotation]) -> List[Annotation]:
    all_entities = sorted(regex_entities + bert_entities, key=lambda x: (x.start, -(x.end - x.start)))
    merged: List[Annotation] = []
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

# ============================================================
# WIKIPEDIA PUBLIC ENTITY CHECK
# ============================================================
wiki_en = wikipediaapi.Wikipedia(user_agent="Privacy_Tool", language="en")
wiki_tr = wikipediaapi.Wikipedia(user_agent="Privacy_Tool", language="tr")

@lru_cache(maxsize=1000)
def is_public_entity(name: str, lang: str) -> bool:
    wiki = wiki_tr if lang == "tr" else wiki_en
    try:
        clean_name = name.lower().strip()
        page = wiki.page(clean_name)
        if not page.exists() or len(page.text) < 500:
            return False
        categories = " ".join(page.categories.keys()).lower()
        strong_signals = [
            "president", "prime minister", "ceo", "founder", "politician",
            "actor", "company", "siyasetçi", "cumhurbaşkanı", "şirket"
        ]
        return any(s in categories for s in strong_signals)
    except:
        return False

# ============================================================
# JURISDICTION SELECTION
# ============================================================
def select_jurisdiction(lang: str) -> str:
    return "kvkk" if lang == "tr" else "gdpr"

# ============================================================
# CONTEXTUAL SCORING
# ============================================================
def contextual_scoring(entity: Annotation, text: str, lang: str) -> Tuple[int, str, List[str]]:
    etype = entity.type.upper()
    val = entity.value
    jurisdiction = select_jurisdiction(lang)

    if etype in ("PERSON", "ORG") and is_public_entity(val, lang):
        return 3, "PASS", ["public_entity_wikipedia"]

    if jurisdiction == "kvkk":
        if etype in ("TCKN", "CREDIT_CARD", "PASSPORT", "IBAN"):
            return 1, "MASK", ["kvkk_special_category"]
        if etype == "EMAIL":
            return (3, "PASS", ["public_email_pattern"]) if is_public_email(val) else (2, "MASK", ["kvkk_personal_data"])
        if etype in ("PERSON", "PHONE", "DATE", "IP"):
            return 2, "MASK", ["kvkk_personal_data"]
        return 3, "PASS", ["kvkk_anonymous"]

    if etype in ("EMAIL", "PHONE", "IP", "CREDIT_CARD"):
        return 1, "MASK", ["gdpr_identifier"]
    if etype == "PERSON":
        return 2, "MASK", ["gdpr_person"]
    return 3, "PASS", ["gdpr_low_risk"]

# ============================================================
# PLACEHOLDER MASKING
# ============================================================
def apply_placeholder_masking(text: str, annotations: List[Annotation]) -> Tuple[str, Dict]:
    counters = defaultdict(int)
    mapping = {}
    display_labels = {"PHONE": "TEL_NUMBER", "DATE": "DATE"}

    for a in annotations:
        if a.action != "MASK":
            continue
        key = (a.type.upper(), a.value)
        if key not in mapping:
            label = display_labels.get(a.type.upper(), a.type.upper())
            counters[label] += 1
            mapping[key] = f"[{label}_{counters[label]}]"
        a.placeholder = mapping[key]

    for a in sorted([x for x in annotations if x.action == "MASK"], key=lambda x: x.start, reverse=True):
        text = text[:a.start] + a.placeholder + text[a.end:]

    return text, mapping

# ============================================================
# SUMMARY TABLE
# ============================================================
def summary_table(annotations: List[Annotation]) -> str:
    lines = ["Value | Type | Level | Action | Reason"]
    for a in annotations:
        lines.append(f"{a.value} | {a.type} | {a.level} | {a.action} | {', '.join(a.reason)}")
    return "\n".join(lines)

# ============================================================
# PDF / DOCX EXPORT
# ============================================================
def export_masked_pdf(text: str) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_file.name, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont(DEFAULT_FONT, 12)
    for line in text.split("\n"):
        c.drawString(50, y, line)
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont(DEFAULT_FONT, 12)
    c.save()
    return temp_file.name

def export_masked_docx(text: str) -> str:
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    doc.save(temp_file.name)
    return temp_file.name

# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================
def analyze_text(text: str, lang: str = "en") -> Tuple[str, str]:
    regex_entities = find_regex_entities(text)
    bert_entities = bert_ner.predict(text, lang)
    merged_entities = merge_entities(regex_entities, bert_entities)

    for e in merged_entities:
        level, action, reason = contextual_scoring(e, text, lang)
        e.level = level
        e.action = action
        e.reason = reason

    masked_text, _ = apply_placeholder_masking(text, merged_entities)
    return summary_table(merged_entities), masked_text

# ============================================================
# STREAMLIT INTERFACE
# ============================================================
st.set_page_config(page_title="PII Masking Tool", layout="wide")
st.title("📄 PII Masking & NER Tool")
st.markdown("PDF/DOCX veya metin dosyalarını okuyarak kişisel verileri maskeleyebilirsiniz.")

uploaded_file = st.file_uploader("Dosya seçin (PDF veya DOCX)", type=["pdf", "docx"])
lang_option = st.selectbox("Dil seçin", ["en", "tr"], index=0)

if uploaded_file:
    file_bytes = uploaded_file.read()
    try:
        if uploaded_file.type == "application/pdf":
            raw_text = read_pdf(file_bytes)
        elif uploaded_file.type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"):
            raw_text = read_docx(file_bytes)
        else:
            st.error("Desteklenmeyen dosya türü.")
            raw_text = ""
    except Exception as e:
        st.error(f"Dosya okunamadı: {e}")
        raw_text = ""

    if raw_text:
        st.subheader("📋 Orijinal Metin")
        st.text_area("Original Text", raw_text, height=300)

        with st.spinner("Metin analiz ediliyor..."):
            summary, masked_text = analyze_text(raw_text, lang_option)

        st.subheader("🔒 Maskelenmiş Metin")
        st.text_area("Masked Text", masked_text, height=300)

        st.subheader("📝 Summary Table")
        st.text_area("Summary Table", summary, height=300)

        pdf_path = export_masked_pdf(masked_text)
        docx_path = export_masked_docx(masked_text)

        st.download_button(
            label="📥 Masked PDF İndir",
            data=open(pdf_path, "rb").read(),
            file_name="masked_output.pdf",
            mime="application/pdf"
        )

        st.download_button(
            label="📥 Masked DOCX İndir",
            data=open(docx_path, "rb").read(),
            file_name="masked_output.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
