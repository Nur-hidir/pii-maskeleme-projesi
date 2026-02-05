import streamlit as st
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# EKSİK OLAN DEĞİŞKEN
PUBLIC_FIGURES = ["Atatürk", "Tarkan", "Elon Musk", "Mustafa Kemal Atatürk"]

# ============================================================
#       MANTIK KISMI (AYNEN KORUNDU)
# ============================================================

REGEX_PATTERNS = {
    "EMAIL": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "PHONE": r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{2,4}",
    "IP": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "TCKN": r"\b\d{11}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
    "DATE_SIMPLE": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"
}

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

class BERTNER:
    def __init__(self):
        self.models = {
            "en": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "tr": "savasy/bert-base-turkish-ner-cased"
        }
        self.pipelines = {}
        for lang, model_name in self.models.items():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.pipelines[lang] = pipeline(
                "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
            )

    def predict(self, text: str, lang: str) -> List[Annotation]:
        ner_results = self.pipelines[lang](text)
        anns = []
        for ent in ner_results:
            anns.append(Annotation(
                type=ent['entity_group'],
                value=ent['word'],
                start=ent['start'],
                end=ent['end'],
                source="bert"
            ))
        return anns

def find_regex_candidates(text: str) -> List[Annotation]:
    res = []
    for label, pattern in REGEX_PATTERNS.items():
        for m in re.finditer(pattern, text):
            res.append(Annotation(
                type=label,
                value=m.group(),
                start=m.start(),
                end=m.end(),
                source="regex"
            ))
    return res

def merge_entities(regex_es: List[Annotation], ml_es: List[Annotation]) -> List[Annotation]:
    all_e = regex_es + ml_es
    all_e.sort(key=lambda x: (x.start, -(x.end - x.start)))
    merged: List[Annotation] = []
    for e in all_e:
        if not merged:
            merged.append(e)
            continue
        last = merged[-1]
        if e.start < last.end:
            priority = {"bert": 3, "regex": 1}
            if priority.get(e.source, 2) > priority.get(last.source, 2):
                merged[-1] = e
            elif priority.get(e.source, 2) == priority.get(last.source, 2):
                if (e.end - e.start) > (last.end - last.start):
                    merged[-1] = e
        else:
            merged.append(e)
    return merged

def contextual_scoring(entity: Annotation, text: str) -> Dict[str, Any]:
    res = {"level": None, "action": None, "reason": []}
    etype = entity.type.upper()
    val = entity.value
    if etype=="PERSON" and val in PUBLIC_FIGURES:
        res.update({"level":3, "action":"PASS"})
        res["reason"].append("public_figure")
        return res
    if etype in ("TCKN","CREDIT_CARD","IP"):
        res.update({"level":1,"action":"MASK"})
        res["reason"].append("explicit_identifier")
        return res
    if etype=="EMAIL":
        if "example" in val.lower() or "test" in val.lower():
            res.update({"level":3,"action":"PASS"})
            res["reason"].append("example_email")
        else:
            res.update({"level":1,"action":"MASK"})
            res["reason"].append("email_contact")
        return res
    if etype=="PHONE":
        res.update({"level":1,"action":"MASK"})
        res["reason"].append("phone_contact")
        return res
    if etype.startswith("DATE"):
        res.update({"level":2,"action":"MASK"})
        res["reason"].append("date_default")
        return res
    if etype in ("ORG","GPE","LOCATION","LOC"):
        res.update({"level":2,"action":"MASK"})
        res["reason"].append("org_location_default")
        return res
    res.update({"level":1,"action":"MASK"})
    res["reason"].append("default")
    return res

class PolicyManager:
    def __init__(self, jurisdiction="gdpr"):
        self.jurisdiction = jurisdiction.lower()
    def decide(self, annotation: Annotation) -> Tuple[str,str]:
        if self.jurisdiction=="gdpr":
            if annotation.level==3: return "PASS","gdpr_low_risk"
            if annotation.level in (1,2): return "MASK","gdpr_mask"
        return "MASK","default_mask"

DEFAULT_EMAIL_MASK = "***@***.com"
DEFAULT_PHONE_MASK = "XXXXXXXXXX"
DEFAULT_DATE_MASK = "XX/XX/XXXX"
DEFAULT_REDACT_MASK = "[REDACTED]"

def apply_masking(text:str, annotations:List[Annotation]) -> str:
    masked = text
    mask_items = sorted([a for a in annotations if a.action=="MASK"], key=lambda x:x.start, reverse=True)
    for a in mask_items:
        mask = DEFAULT_REDACT_MASK
        if a.type=="EMAIL": mask = DEFAULT_EMAIL_MASK
        elif a.type=="PHONE": mask = DEFAULT_PHONE_MASK
        elif a.type.startswith("DATE"): mask = DEFAULT_DATE_MASK
        s, e = a.start, a.end
        masked = masked[:s] + mask + masked[e:]
    return masked

def summary_table(annotations:List[Annotation]) -> str:
    lines = ["Value | Type | Level | Action | Reason"]
    for a in annotations:
        reason_str = ", ".join(a.reason) if a.reason else "N/A"
        lines.append(f"{a.value} | {a.type} | {a.level} | {a.action} | {reason_str}")
    return "\n".join(lines)

def analyze_text(text:str, lang:str="en") -> Tuple[str,str]:
    bert_ner = BERTNER()
    regex_candidates = find_regex_candidates(text)
    ml_entities = bert_ner.predict(text, lang)
    merged = merge_entities(regex_candidates, ml_entities)
    policy_mgr = PolicyManager("gdpr")
    annotations = []
    for e in merged:
        score = contextual_scoring(e, text)
        e.level = score["level"]
        e.reason = score["reason"]
        e.action, pol = policy_mgr.decide(e)
        e.reason.append(pol)
        annotations.append(e)
    masked_text = apply_masking(text, annotations)
    table = summary_table(annotations)
    return table, masked_text

# ============================================================
#       ARAYÜZ AYARLARI (GÜNCELLENDİ: YEŞİL BUTON)
# ============================================================

st.set_page_config(page_title="PII Maskeleme", layout="wide")

# --- CSS İLE BUTONU YEŞİL YAPMA KODU ---
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #28a745; /* Yeşil Renk Kodu */
        color: white;
        border: none;
        font-size: 18px;
        padding: 10px 24px;
    }
    div.stButton > button:first-child:hover {
        background-color: #218838; /* Üzerine gelince koyu yeşil */
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
# ---------------------------------------

st.title("PII Maskeleme") 

with st.sidebar:
    st.header("Ayarlar")
    lang_choice = st.selectbox(
        "Dil Seçiniz",
        options=["en", "tr"],
        format_func=lambda x: "Türkçe (tr)" if x == "tr" else "English (en)"
    )

st.write("Aşağıdaki kutuya metninizi girin ve analiz butonuna basın.")

text_input = st.text_area("Sadece metin girin", height=150)

# Buton (Artık CSS sayesinde Yeşil olacak)
if st.button("Analiz Et"):
    if not text_input.strip():
        st.warning("Lütfen boş bırakmayınız.")
    else:
        with st.spinner("Modeller yükleniyor..."):
            try:
                table_result, masked_result = analyze_text(text_input, lang_choice)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Orijinal Metin")
                    st.info(text_input)
                with col2:
                    st.subheader("Maskelenmiş Metin")
                    st.success(masked_result)
                
                st.subheader("Analiz Tablosu")
                st.text(table_result)
                
            except Exception as e:
                st.error(f"Hata: {e}")
