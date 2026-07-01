# 🛡️ PII Masking Studio: Hibrit Veri Anonimleştirme Motoru

PII Masking Studio; metinler, dökümanlar ve görseller üzerinde yer alan ve KVKK kapsamına giren hassas verileri (PII - Kişisel Veri) akıllıca tespit edip anonimleştiren bir veri güvenliği projesidir. 

Bu projeyi geliştirirken amacım, geleneksel kural tabanlı (Regex) yöntemlerin kaçırdığı yerleri derin öğrenme (BERT NER) modelleriyle yakalamak ve ikisini esnek bir **hibrit mimaride** bir araya getirmekti.

---

##  Öne Çıkan Özellikler

*   **Çift Motorlu Hibrit Tespit:** Geleneksel Regex kuralları ile `savasy/bert-base-turkish-ner-cased` BERT modelini eş zamanlı çalıştırarak veri kaçırma ihtimalini minimuma indirdim.
*   **Çok Yönlü Girdi Desteği (Multi-modal):** Sadece düz metinleri değil; PDF ve DOCX dosyalarını, hatta Tesseract OCR entegrasyonu sayesinde resimlerdeki (PNG, JPG) metinleri bile okuyup analiz edebiliyor.
*   **Akıllı Bağlamsal Skorlama (Contextual Scoring):** Yakalanan her varlığı doğrudan sansürlemek yerine; harf duyarlılığı, veri türü önceliği ve KVKK'nın özel nitelikli veri sınıflarını göz önünde bulunduran dinamik bir puanlama algoritması kurguladım.
*   **Gereksiz Maskeleme Filtresi (Wikipedia API):** Kamuya mal olmuş (Wikipedia'da kayıtlı) tanınmış kişi veya kurum isimlerini otomatik olarak ayırt ediyor, böylece her özel ismin gereksiz yere maskelenmesini (False Positive) engelliyor.
*   **Metin Bütünlüğünü Koruyan Maskeleme:** Verileri tamamen silip metni okunmaz hale getirmek yerine, akışın bozulmaması için `[PERSON_1]`, `[TEL_NUMBER_1]` gibi akıllı ve birbiriyle eşleşen etiketler yerleştiriyor.
*   **Gelişmiş Dışa Aktarım:** Maskelenen verileri arayüzde hem metin hem de tablo olarak anlık raporlarken; çıktıları Türkçe karakter sorunu yaşamadan (Unicode destekli) PDF veya DOCX formatında indirme imkanı sunuyor.

---

##  Teknolojik Altyapı

Projeyi tamamen modüler ve modern veri bilimi kütüphanelerini temel alarak inşa ettim:

*   **Arayüz:** Streamlit (Kullanıcı deneyimini artırmak için özel karanlık tema ve CSS dokunuşlarıyla özelleştirildi)
*   **Doğal Dil İşleme (NLP):** Hugging Face Transformers & PyTorch (BERT tabanlı Varlık Tanıma)
*   **Görsel İşleme & OCR:** Tesseract OCR, OpenCV ve Pillow (Görselleri netleştirip yazıya dökme)
*   **Döküman Yönetimi:** `pdfplumber` ve `python-docx`
*   **Raporlama:** ReportLab (Metin kaymalarını önleyen otomatik satır yönetimi ve özel font kayıt mimarisi ile)

---

##  Tespit Edilen Veri Sınıfları

| Veri Türü | Tespit Yöntemi | KVKK Yaklaşımı |
| :--- | :--- | :--- |
| **TCKN, IBAN, Kredi Kartı, Pasaport** | Regex / Öncelikli Eşleşme | Özel Nitelikli Veri / Kesin Maskeleme |
| **Telefon, E-posta, IP Adresi, Tarih** | Regex / Bağlamsal Analiz | Kişisel Veri / Maskeleme |
| **Kişi (PER), Konum (LOC), Kurum (ORG)**| BERT NER / Wikipedia Kontrolü| Kamusal Değilse Maskeleme |

---

##  Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda denemek isterseniz:

### 1. Kütüphanelerin Yüklenmesi
Sisteminizde Python 3.8+ ve Tesseract OCR'ın kurulu olduğundan emin olduktan sonra terminalde şu komutu çalıştırabilirsiniz:

```bash
pip install streamlit transformers torch pdfplumber python-docx reportlab pytesseract wikipedia-api numpy opencv-python pandas pillow
