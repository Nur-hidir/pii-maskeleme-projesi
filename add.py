import streamlit as st
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ----------------------------
# 1. AYARLAR (STREAMLIT İÇİN GEREKLİ)
# ----------------------------
st.set_page_config(page_title="PII Maskeleme", layout="wide")

# CSS ile Butonu Yeşil Yapma
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #28a745;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #218838;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# 2. SENİN BELİRLEDİĞİN REGEX KURALLARI (AYNEN KORUNDU)
# ----------------------------
REGEX_PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "PHONE": r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{2,4}",
    "IP": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "TCKN": r"\b\d{11}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
    "DATE_SIMPLE": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    "ID": r"\b(?:ID|id|Id)[\s:.-]?[A-Za-z0-9]{3,}\b|\b[A-Z]{2,}\d{3,}\b|\b\d{3,}[A-Z]{2,}\b"
}

# ----------------------------
# Public Figures List
# ----------------------------
PUBLIC_FIGURES = ["Elon Musk", "Barack Obama", "Angela Merkel"]

# ----------------------------
# Dataclass
# ----------------------------
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

# ----------------------------
# BERT NER Pipeline
# ----------------------------
class BERTNER:
    def __init__(self):
        self.models = {
            "en": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "tr": "savasy/bert-base-turkish-ner-cased"
        }
        self.pipelines = {}
        
        # STREAMLIT İÇİN CACHE EKLENDİ (Performans için şart, mantığı değiştirmez)
        @st.cache_resource
        def load_pipeline(model_name):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

        for lang, model_name in self.models.items():
            self.pipelines[lang] = load_pipeline(model_name)

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

# ----------------------------
# Regex detection
# ----------------------------
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

# ----------------------------
# Merge Entities (SENİN YAZDIĞIN KORUMA MANTIĞI)
# ----------------------------
def merge_entities(regex_es: List[Annotation], ml_es: List[Annotation]) -> List[Annotation]:
    # EMAIL'leri kesin korumak için özel işlem
    emails = [e for e in regex_es if e.type == "EMAIL"]
    ids = [e for e in regex_es if e.type == "ID"]
    
    # Önce email ve ID'lerin span'lerini belirle
    protected_spans = [(e.start, e.end) for e in emails + ids]
    
    # ML entitylerini filtrele: email/id span'leriyle çakışan ML entityleri çıkar
    filtered_ml = []
    for ml_e in ml_es:
        overlap = False
        for ps, pe in protected_spans:
            if not (ml_e.end <= ps or ml_e.start >= pe):
                overlap = True
                break
        if not overlap:
            filtered_ml.append(ml_e)
    
    # Diğer regex entityleri
    other_regex = [e for e in regex_es if e.type not in ("EMAIL", "ID")]
    
    # Tüm entityleri birleştir: EMAIL ve ID'ler önce
    all_e = emails + ids + other_regex + filtered_ml
    all_e.sort(key=lambda x: (x.start, -(x.end - x.start)))

    merged: List[Annotation] = []
    for e in all_e:
        if not merged:
            merged.append(e)
            continue

        last = merged[-1]
        # Overlap kontrolü
        if e.start < last.end:
            # Eğer son eklenen EMAIL veya ID ise, kesinlikle koru
            if last.type in ("EMAIL", "ID"):
                continue
            # Eğer yeni gelen EMAIL veya ID ise, kesinlikle tercih et
            elif e.type in ("EMAIL", "ID"):
                merged[-1] = e
            else:
                priority = {"regex": 3, "bert": 2}
                if priority.get(e.source, 1) > priority.get(last.source, 1):
                    merged[-1] = e
                elif priority.get(e.source, 1) == priority.get(last.source, 1):
                    if (e.end - e.start) > (last.end - last.start):
                        merged[-1] = e
        else:
            merged.append(e)
    return merged

# ----------------------------
# Contextual Scoring (SENİN KODUN)
# ----------------------------
def contextual_scoring(entity: Annotation, text: str) -> Dict[str, Any]:
    res = {"level": None, "action": None, "reason": []}
    etype = entity.type.upper()
    val = entity.value

    if etype=="PERSON" and val in PUBLIC_FIGURES:
        res.update({"level":3, "action":"PASS"})
        res["reason"].append("public_figure")
        return res

    if etype in ("TCKN","CREDIT_CARD","IP","ID"):
        res.update({"level":1,"action":"MASK"})
        res["reason"].append("explicit_identifier")
        return res

    if etype=="EMAIL":
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

# ----------------------------
# Policy Manager
# ----------------------------
class PolicyManager:
    def __init__(self, jurisdiction="gdpr"):
        self.jurisdiction = jurisdiction.lower()

    def decide(self, annotation: Annotation) -> Tuple[str,str]:
        if self.jurisdiction=="gdpr":
            if annotation.level==3: return "PASS","gdpr_low_risk"
            if annotation.level in (1,2): return "MASK","gdpr_mask"
        return "MASK","default_mask"

# ----------------------------
# Masking Logic (SENİN PARÇALI MASKELEME MANTIĞIN)
# ----------------------------
DEFAULT_EMAIL_MASK = "***@***.com"
DEFAULT_PHONE_MASK = "XXXXXXXXXX"
DEFAULT_DATE_MASK = "XX/XX/XXXX"
DEFAULT_ID_MASK = "[ID-REDACTED]"
DEFAULT_REDACT_MASK = "[REDACTED]"

def apply_masking(text:str, annotations:List[Annotation]) -> str:
    masked = text
    mask_items = sorted([a for a in annotations if a.action=="MASK"], key=lambda x:x.start, reverse=True)
    for a in mask_items:
        mask = DEFAULT_REDACT_MASK
        if a.type=="EMAIL":
            email_val = a.value
            if "@" in email_val:
                parts = email_val.split("@")
                user_part = parts[0]
                domain_part = parts[1]
                
                if len(user_part) >= 1:
                    masked_user = user_part[0] + "*" * max(1, len(user_part) - 1)
                else:
                    masked_user = "*"
                
                if len(domain_part) >= 4:
                    masked_domain = "*" * (len(domain_part) - 4) + domain_part[-4:]
                else:
                    masked_domain = "*" * len(domain_part)
                
                mask = masked_user + "@" + masked_domain
            else:
                mask = DEFAULT_EMAIL_MASK
        elif a.type=="PHONE": 
            mask = DEFAULT_PHONE_MASK
        elif a.type.startswith("DATE"): 
            mask = DEFAULT_DATE_MASK
        elif a.type=="ID": 
            mask = DEFAULT_ID_MASK

        s,e = a.start, a.end
        masked = masked[:s] + mask + masked[e:]
    return masked

# ----------------------------
# Summary Table
# ----------------------------
def summary_table(annotations:List[Annotation]) -> str:
    lines = ["Value | Type | Level | Action | Reason"]
    for a in annotations:
        reason_str = ", ".join(a.reason) if a.reason else "N/A"
        lines.append(f"{a.value} | {a.type} | {a.level} | {a.action} | {reason_str}")
    return "\n".join(lines)

# ----------------------------
# Main Function
# ----------------------------
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
#       3. STREAMLIT ARAYÜZÜ (BURASI DEĞİŞTİ, MANTIK AYNI)
# ============================================================

st.title("PII Maskeleme") 

with st.sidebar:
    st.header("Ayarlar")
    lang_choice = st.selectbox(
        "Dil Seçiniz",
        options=["en", "tr"],
        format_func=lambda x: "Türkçe (tr)" if x == "tr" else "English (en)"
    )

st.write("Aşağıdaki kutuya metninizi girin ve analiz butonuna basın.")

# Streamlit Text Area (Senin widgets.Textarea yerine)
text_input = st.text_area("Sadece metin girin", height=150)

# Streamlit Buton (Senin widgets.Button yerine)
if st.button("Analiz Et"):
    if not text_input.strip():
        st.warning("Lütfen boş bırakmayınız.")
    else:
        with st.spinner("Analiz yapılıyor..."):
            try:
                # Senin fonksiyonunu çağırıyoruz
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
