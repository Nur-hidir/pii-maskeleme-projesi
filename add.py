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

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
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
# FONT (UNICODE SUPPORT)
# ============================================================

def register_fonts():
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))
        return "DejaVuSans"
    return "Helvetica"


DEFAULT_FONT = register_fonts()


# ============================================================
# FILE READERS
# ============================================================

def read_pdf(file_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
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
# BERT NER
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
                    "ner",
                    model=AutoModelForTokenClassification.from_pretrained(model),
                    tokenizer=AutoTokenizer.from_pretrained(model),
                    aggregation_strategy="max"
                )

    def predict(self, text: str, lang: str):

        return [
            Annotation(
                type=e["entity_group"],
                value=e["word"],
                start=e["start"],
                end=e["end"],
                source="bert"
            )
            for e in self._pipes[lang](text)
        ]


bert_ner = BERTNER()


# ============================================================
# REGEX DETECTION
# ============================================================

def find_regex_entities(text: str):

    entities = []

    for label, pattern in REGEX_PATTERNS.items():

        for m in re.finditer(pattern, text):

            entities.append(
                Annotation(label, m.group(), m.start(), m.end(), "regex")
            )

    return entities


# ============================================================
# MERGE ENTITIES
# ============================================================

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


# ============================================================
# WIKIPEDIA PUBLIC ENTITY CHECK
# ============================================================

wiki_en = wikipediaapi.Wikipedia(user_agent="privacy_tool", language="en")
wiki_tr = wikipediaapi.Wikipedia(user_agent="privacy_tool", language="tr")


@lru_cache(maxsize=1000)
def is_public_entity(name, lang):

    wiki = wiki_tr if lang == "tr" else wiki_en

    try:

        page = wiki.page(name)

        if not page.exists() or len(page.text) < 500:
            return False

        categories = " ".join(page.categories.keys()).lower()

        signals = [
            "president",
            "prime minister",
            "ceo",
            "founder",
            "politician",
            "actor",
            "company",
            "siyasetçi",
            "cumhurbaşkanı",
            "şirket"
        ]

        return any(s in categories for s in signals)

    except:
        return False


# ============================================================
# CONTEXTUAL SCORING
# ============================================================

def contextual_scoring(entity, lang):

    etype = entity.type.upper()
    val = entity.value

    if etype in ("PERSON", "ORG") and is_public_entity(val, lang):
        return 3, "PASS", ["public_entity"]

    if lang == "tr":

        if etype in ("TCKN", "CREDIT_CARD", "PASSPORT", "IBAN"):
            return 1, "MASK", ["kvkk_sensitive"]

        if etype in ("PERSON", "PHONE", "EMAIL", "DATE", "IP"):
            return 2, "MASK", ["kvkk_personal"]

    else:

        if etype in ("EMAIL", "PHONE", "IP", "CREDIT_CARD"):
            return 1, "MASK", ["gdpr_identifier"]

        if etype == "PERSON":
            return 2, "MASK", ["gdpr_person"]

    return 3, "PASS", ["low_risk"]


# ============================================================
# MASKING
# ============================================================

def apply_placeholder_masking(text, annotations):

    counters = defaultdict(int)
    mapping = {}

    for a in annotations:

        if a.action != "MASK":
            continue

        key = (a.type, a.value)

        if key not in mapping:

            counters[a.type] += 1

            mapping[key] = f"[{a.type}_{counters[a.type]}]"

        a.placeholder = mapping[key]

    for a in sorted([x for x in annotations if x.action == "MASK"], key=lambda x: x.start, reverse=True):

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
# PDF EXPORT (FIXED FOR TURKISH)
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


# ============================================================
# DOCX EXPORT
# ============================================================

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

    masked = apply_placeholder_masking(text, merged)

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
        raw_text = read_pdf(file_bytes)

    else:
        raw_text = read_docx(file_bytes)

    st.subheader("Original Text")

    st.text_area("", raw_text, height=250)

    summary, masked_text = analyze_text(raw_text, lang)

    st.subheader("Masked Text")

    st.text_area("", masked_text, height=250)

    st.subheader("Summary")

    st.text_area("", summary, height=250)

    pdf_path = export_masked_pdf(masked_text)

    docx_path = export_masked_docx(masked_text)

    st.download_button(
        "Download Masked PDF",
        data=open(pdf_path, "rb").read(),
        file_name="masked_output.pdf"
    )

    st.download_button(
        "Download Masked DOCX",
        data=open(docx_path, "rb").read(),
        file_name="masked_output.docx"
    )
