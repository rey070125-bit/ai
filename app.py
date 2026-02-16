from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pytesseract
import pdfplumber
from PIL import Image
import docx
import tempfile
from pytesseract import Output


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://e201filems.infinityfree.me"]}})

@app.get("/health")
def health():
    return "ok", 200
# ==================================================
# OCR QUALITY CHECK (NEW)
# ==================================================

def check_image_readability(file_path):
    try:
        image = Image.open(file_path).convert("RGB")

        # Get OCR data with confidence scores
        data = pytesseract.image_to_data(image, output_type=Output.DICT)

        confidences = []
        for conf in data.get("conf", []):
            try:
                c = float(conf)
                if c >= 0:
                    confidences.append(c)
            except:
                pass

        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        normalized_conf = round(avg_conf / 100, 2)

        text = pytesseract.image_to_string(image)
        text_length = len("".join(ch for ch in text if ch.isalnum()))

        readable = True
        reason = "ok"

        if normalized_conf < 0.45:
            readable = False
            reason = "low_ocr_confidence"
        elif text_length < 25:
            readable = False
            reason = "too_little_text"

        return {
            "readable": readable,
            "ocr_confidence": normalized_conf,
            "text_length": text_length,
            "quality_reason": reason
        }

    except Exception as e:
        return {
            "readable": False,
            "ocr_confidence": 0,
            "text_length": 0,
            "quality_reason": "ocr_processing_error"
        }

# ==================================================
# TEXT EXTRACTION
# ==================================================

def extract_text(file_path, ext):
    text = ""

    try:
        if ext == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""

        elif ext == ".docx":
            doc = docx.Document(file_path)
            text = " ".join(p.text for p in doc.paragraphs)

        elif ext in [".jpg", ".jpeg", ".png"]:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)

        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

    except Exception as e:
        print("Text extraction error:", e)

    return text.lower()

# ==================================================
# CONTENT-BASED CLASSIFICATION
# ==================================================

def classify_by_content(text):
    rules = {
        "resume": ["education", "skills", "work experience", "objective", "personal data"],
        "birth_cert": ["certificate of live birth", "date of birth", "place of birth"],
        "tin": ["tin", "bureau of internal revenue", "tax identification number"],
        "sss": ["sss number", "social security system"],
        "philhealth": ["philhealth", "insurance corporation"],
        "pagibig": ["pag-ibig", "hdmf"],
        "contract": ["agreement", "terms and conditions", "shall"],
        "medical_clearance": ["medical clearance", "fit to work"],
        "memo": ["memorandum", "subject:"],
        "incident_report": ["incident", "incident occurred"],
        "disciplinary_action": ["disciplinary action", "violation"],
        "commendation": ["commendation", "outstanding performance"],
        "exit_letter": ["resignation", "last working day"],
        "interview": ["exit interview"],
        "clearance": ["clearance form", "no pending accountability"]
    }

    scores = {}
    for doc_type, keywords in rules.items():
        scores[doc_type] = sum(text.count(k) for k in keywords)

    best_match = max(scores, key=scores.get)

    if scores[best_match] == 0:
        return "others", 0.55

    confidence = min(0.95, 0.6 + scores[best_match] * 0.05)
    return best_match, confidence

# ==================================================
# API ENDPOINT
# ==================================================

@app.route('/classify', methods=['POST'])
def classify():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file"}), 400

    ext = os.path.splitext(file.filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp:
        file.save(temp.name)
        temp_path = temp.name

    # NEW: readability check for images
    readability_data = {
        "readable": True,
        "ocr_confidence": None,
        "text_length": None,
        "quality_reason": "not_image"
    }

    if ext in [".jpg", ".jpeg", ".png"]:
        readability_data = check_image_readability(temp_path)

    text = extract_text(temp_path, ext)
    os.remove(temp_path)

    if not text.strip():
        return jsonify({
            "document_type": "others",
            "confidence": 0.50,
            **readability_data
        })

    doc_type, confidence = classify_by_content(text)

    return jsonify({
        "document_type": doc_type,
        "confidence": round(confidence, 2),
        **readability_data
    })

# ==================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

