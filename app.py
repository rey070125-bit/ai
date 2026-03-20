from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import json
import tempfile
import pytesseract
import pdfplumber
from PIL import Image
import docx
from pytesseract import Output

# ==================================================
# OPTIONAL TESSERACT BINARY PATH
# - Local Windows: set TESSERACT_CMD env if needed
# - Render/Linux: usually leave unset if tesseract is in PATH
# ==================================================
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

app = Flask(__name__)

# ==================================================
# CORS
# ==================================================
DEFAULT_ORIGINS = [
    "https://e201filems.infinityfree.me",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

cors_origins_env = os.getenv("CORS_ORIGINS", "").strip()
if cors_origins_env:
    ALLOWED_ORIGINS = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
else:
    ALLOWED_ORIGINS = DEFAULT_ORIGINS

CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# ==================================================
# OCR SETTINGS
# ==================================================
OCR_TIMEOUT = int(os.getenv("OCR_TIMEOUT", "8"))
OCR_CONFIG = os.getenv("OCR_CONFIG", "--oem 1 --psm 6")
MAX_DIM = int(os.getenv("OCR_MAX_DIM", "1600"))

# ==================================================
# HEALTH CHECK
# ==================================================
@app.get("/health")
def health():
    return "ok", 200

# ==================================================
# OCR HELPERS
# ==================================================
def prep_image_for_ocr(file_path: str) -> Image.Image:
    """
    Open, normalize, grayscale, and resize image for faster OCR.
    """
    img = Image.open(file_path)

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    img = img.convert("L")

    w, h = img.size
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img

def safe_ocr_string(img: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(img, config=OCR_CONFIG, timeout=OCR_TIMEOUT)
    except RuntimeError:
        return ""
    except pytesseract.pytesseract.TesseractError:
        return ""

# ==================================================
# OCR QUALITY CHECK
# ==================================================
def check_image_readability(file_path):
    try:
        image = prep_image_for_ocr(file_path)

        try:
            data = pytesseract.image_to_data(
                image,
                output_type=Output.DICT,
                config=OCR_CONFIG,
                timeout=OCR_TIMEOUT
            )
        except RuntimeError:
            return {
                "readable": False,
                "ocr_confidence": 0,
                "text_length": 0,
                "quality_reason": "ocr_timeout"
            }
        except pytesseract.pytesseract.TesseractError:
            return {
                "readable": False,
                "ocr_confidence": 0,
                "text_length": 0,
                "quality_reason": "tesseract_error"
            }

        confidences = []
        for conf in data.get("conf", []):
            try:
                c = float(conf)
                if c >= 0:
                    confidences.append(c)
            except Exception:
                pass

        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        normalized_conf = round(avg_conf / 100, 2)

        text = safe_ocr_string(image)
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

    except Exception:
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
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"

        elif ext == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join(p.text for p in doc.paragraphs)

        elif ext in [".jpg", ".jpeg", ".png"]:
            image = prep_image_for_ocr(file_path)
            text = safe_ocr_string(image)

        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

    except Exception as e:
        print("Text extraction error:", e)

    return text

# ==================================================
# TEXT NORMALIZATION
# ==================================================
def normalize_text(text):
    text = (text or "").lower()

    replacements = {
        "pag ibig": "pag-ibig",
        "pagibig": "pag-ibig",
        "phil health": "philhealth",
        "n.b.i": "nbi",
        "b.i.r": "bir",
        "&": " and ",
        "\n": " ",
        "\r": " ",
        "\t": " "
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^a-z0-9\s\-\/:,\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ==================================================
# MATCH HELPERS
# ==================================================
def phrase_exists(text, phrase):
    return phrase in text

def count_phrase(text, phrase):
    return text.count(phrase)

def compute_rule_score(text, rule):
    score = 0
    matched_terms = []

    for phrase, weight in rule.get("strong", {}).items():
        hits = count_phrase(text, phrase)
        if hits > 0:
            score += hits * weight
            matched_terms.append(phrase)

    for phrase, weight in rule.get("medium", {}).items():
        hits = count_phrase(text, phrase)
        if hits > 0:
            score += hits * weight
            matched_terms.append(phrase)

    for phrase, weight in rule.get("weak", {}).items():
        hits = count_phrase(text, phrase)
        if hits > 0:
            score += hits * weight
            matched_terms.append(phrase)

    return score, matched_terms

# ==================================================
# CONTENT-BASED CLASSIFICATION
# ==================================================
def classify_by_content(raw_text):
    text = normalize_text(raw_text)

    rules = {
        "nbi_clearance": {
            "strong": {
                "nbi clearance": 8,
                "national bureau of investigation": 8
            },
            "medium": {
                "nbi": 4,
                "no criminal record": 4,
                "no derogatory record": 4
            },
            "weak": {},
            "requires_any": [
                "nbi clearance",
                "national bureau of investigation",
                "nbi"
            ],
            "min_score": 8
        },

        "barangay_clearance": {
            "strong": {
                "barangay clearance": 8,
                "office of the punong barangay": 7,
                "punong barangay": 6
            },
            "medium": {
                "barangay captain": 4,
                "barangay certificate": 4,
                "office of the barangay captain": 4
            },
            "weak": {
                "barangay": 2
            },
            "requires_any": [
                "barangay clearance",
                "punong barangay",
                "barangay captain",
                "barangay"
            ],
            "min_score": 8
        },

        "medical_clearance": {
            "strong": {
                "medical clearance": 8,
                "fit to work": 7
            },
            "medium": {
                "physician": 3,
                "clinic": 3,
                "medical certificate": 4,
                "medically fit": 5
            },
            "weak": {
                "medical": 1
            },
            "requires_any": [
                "medical clearance",
                "fit to work",
                "medical certificate",
                "medically fit"
            ],
            "min_score": 7
        },

        "birth_cert": {
            "strong": {
                "certificate of live birth": 9
            },
            "medium": {
                "date of birth": 3,
                "place of birth": 3,
                "registry number": 3,
                "live birth": 5
            },
            "weak": {
                "birth": 1
            },
            "requires_any": [
                "certificate of live birth",
                "live birth"
            ],
            "min_score": 8
        },

        "tin": {
            "strong": {
                "tax identification number": 8,
                "bureau of internal revenue": 7
            },
            "medium": {
                "tin": 4,
                "bir": 4
            },
            "weak": {},
            "requires_any": [
                "tax identification number",
                "bureau of internal revenue",
                "tin",
                "bir"
            ],
            "min_score": 7
        },

        "sss": {
            "strong": {
                "unified multi-purpose id": 8
            },
            "medium": {
                "social security system": 6,
                "crn": 4,
                "sss number": 6,
                "sss": 4
            },
            "weak": {},
            "requires_any": [
                "social security system",
                "sss number",
                "sss",
                "unified multi-purpose id"
            ],
            "min_score": 7
        },

        "philhealth": {
            "strong": {
                "philhealth": 8,
                "philippine health insurance corporation": 8
            },
            "medium": {
                "member data record": 4
            },
            "weak": {},
            "requires_any": [
                "philhealth",
                "philippine health insurance corporation"
            ],
            "min_score": 7
        },

        "pagibig": {
            "strong": {
                "pag-ibig": 8,
                "home development mutual fund": 8
            },
            "medium": {
                "hdmf": 5,
                "pag ibig": 5
            },
            "weak": {},
            "requires_any": [
                "pag-ibig",
                "home development mutual fund",
                "hdmf"
            ],
            "min_score": 7
        },

        "resume": {
            "strong": {
                "curriculum vitae": 8,
                "resume": 7
            },
            "medium": {
                "work experience": 4,
                "educational background": 4,
                "skills": 3,
                "objective": 3,
                "personal information": 3,
                "education": 2
            },
            "weak": {},
            "requires_any": [
                "curriculum vitae",
                "resume",
                "work experience",
                "educational background",
                "education"
            ],
            "min_score": 7
        },

        "contract": {
            "strong": {
                "employment contract": 8,
                "contract of employment": 8,
                "this agreement": 6
            },
            "medium": {
                "terms and conditions": 4,
                "employee": 1,
                "employer": 2,
                "agreement": 3,
                "shall": 1
            },
            "weak": {},
            "requires_any": [
                "employment contract",
                "contract of employment",
                "this agreement",
                "terms and conditions"
            ],
            "min_score": 7
        },

        "memo": {
            "strong": {
                "memorandum": 8
            },
            "medium": {
                "subject": 2,
                "to ": 1,
                "from ": 1
            },
            "weak": {},
            "requires_any": [
                "memorandum"
            ],
            "min_score": 6
        },

        "incident_report": {
            "strong": {
                "incident report": 8
            },
            "medium": {
                "incident occurred": 4,
                "date of incident": 4,
                "reported by": 2
            },
            "weak": {
                "incident": 1
            },
            "requires_any": [
                "incident report",
                "incident occurred",
                "date of incident"
            ],
            "min_score": 6
        },

        "disciplinary_action": {
            "strong": {
                "disciplinary action": 8,
                "notice to explain": 7
            },
            "medium": {
                "violation": 3,
                "misconduct": 3,
                "warning": 2
            },
            "weak": {},
            "requires_any": [
                "disciplinary action",
                "notice to explain",
                "violation"
            ],
            "min_score": 6
        },

        "commendation": {
            "strong": {
                "certificate of commendation": 8,
                "commendation": 7
            },
            "medium": {
                "outstanding performance": 4,
                "recognition": 3
            },
            "weak": {},
            "requires_any": [
                "commendation",
                "certificate of commendation",
                "outstanding performance"
            ],
            "min_score": 6
        },

        "exit_letter": {
            "strong": {
                "resignation letter": 8
            },
            "medium": {
                "resignation": 4,
                "last working day": 4,
                "effective immediately": 3
            },
            "weak": {},
            "requires_any": [
                "resignation letter",
                "resignation",
                "last working day"
            ],
            "min_score": 6
        },

        "interview": {
            "strong": {
                "exit interview": 8
            },
            "medium": {
                "interview": 2,
                "separation feedback": 4
            },
            "weak": {},
            "requires_any": [
                "exit interview",
                "separation feedback"
            ],
            "min_score": 6
        },

        "clearance": {
            "strong": {
                "clearance form": 7,
                "no pending accountability": 7,
                "employee clearance": 7
            },
            "medium": {
                "clearance": 2,
                "accountability": 3
            },
            "weak": {},
            "requires_any": [
                "clearance form",
                "no pending accountability",
                "employee clearance",
                "clearance"
            ],
            "min_score": 6
        }
    }

    priority_order = [
        "nbi_clearance",
        "barangay_clearance",
        "medical_clearance",
        "birth_cert",
        "tin",
        "sss",
        "philhealth",
        "pagibig",
        "resume",
        "contract",
        "memo",
        "incident_report",
        "disciplinary_action",
        "commendation",
        "exit_letter",
        "interview",
        "clearance"
    ]

    best_doc = "others"
    best_score = 0

    for doc_type in priority_order:
        rule = rules[doc_type]

        if not any(phrase_exists(text, req) for req in rule.get("requires_any", [])):
            continue

        score, _matches = compute_rule_score(text, rule)

        if score >= rule.get("min_score", 1) and score > best_score:
            best_doc = doc_type
            best_score = score

    if best_doc == "others":
        return "others", 0.55

    if best_score >= 18:
        confidence = 0.95
    elif best_score >= 14:
        confidence = 0.90
    elif best_score >= 10:
        confidence = 0.85
    elif best_score >= 8:
        confidence = 0.80
    else:
        confidence = 0.70

    return best_doc, round(confidence, 2)

# ==================================================
# VERIFICATION HELPERS
# ==================================================
def parse_request_json_field(name, default=None):
    raw = request.form.get(name)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default

def normalize_person_name(value):
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z\s\-\.]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value

def tokens_from_name(value):
    norm = normalize_person_name(value)
    return [t for t in norm.split(" ") if len(t) >= 2]

def name_appears_in_text(name, text):
    if not name or not text:
        return False

    text_norm = normalize_text(text)
    tokens = tokens_from_name(name)

    if len(tokens) >= 2:
        hits = sum(1 for t in tokens if t in text_norm)
        return hits >= 2

    return normalize_person_name(name) in text_norm

def extract_date_candidates(text):
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,15}\s+\d{4}\b",
        r"\b[A-Za-z]{3,15}\s+\d{1,2},\s+\d{4}\b",
    ]

    found = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text, flags=re.IGNORECASE))

    unique = []
    seen = set()
    for item in found:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:10]

def extract_license_number(text):
    patterns = [
        r"\b(?:lic(?:ense)?|license|prc)\s*(?:no\.?|number|#)?\s*[:\-]?\s*([a-z0-9\-]{5,20})\b",
        r"\b([0-9]{5,12})\b"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def extract_doctor_name(text):
    patterns = [
        r"\bdr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b",
        r"\bdoctor\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
    return None

def extract_clinic_name(text):
    patterns = [
        r"\b[A-Z][A-Za-z0-9&,\.\-\s]{3,80}\s+Clinic\b",
        r"\b[A-Z][A-Za-z0-9&,\.\-\s]{3,80}\s+Hospital\b",
        r"\b[A-Z][A-Za-z0-9&,\.\-\s]{3,80}\s+Medical Center\b"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
    return None

def extract_resume_features(raw_text):
    text = normalize_text(raw_text)
    features = {
        "has_work_experience": "work experience" in text,
        "has_education": "educational background" in text or "education" in text,
        "has_skills": "skills" in text,
        "has_objective": "objective" in text or "career objective" in text,
        "has_email": bool(re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}", raw_text)),
        "has_phone": bool(re.search(r"(09\d{9}|\+639\d{9})", raw_text)),
        "date_candidates": extract_date_candidates(raw_text)
    }
    return features

def extract_medical_clearance_fields(raw_text):
    return {
        "employee_name_in_text": None,
        "doctor_name": extract_doctor_name(raw_text),
        "clinic_name": extract_clinic_name(raw_text),
        "license_number": extract_license_number(raw_text),
        "date_candidates": extract_date_candidates(raw_text)
    }

def push_flag(flags, code, message, severity="low"):
    flags.append({
        "code": code,
        "message": message,
        "severity": severity
    })

def score_resume_risk(raw_text, employee_context, detected_type, readability_data):
    text = normalize_text(raw_text)
    flags = []
    risk = 0

    employee_name = (employee_context or {}).get("employee_name", "")
    email = (employee_context or {}).get("email", "")
    contact_no = (employee_context or {}).get("contact_no", "")

    features = extract_resume_features(raw_text)

    if detected_type not in ["resume", "others"]:
        push_flag(flags, "type_mismatch", f"Detected type is '{detected_type}' instead of 'resume'.", "high")
        risk += 35

    if len(text) < 120:
        push_flag(flags, "too_little_text", "Resume contains too little readable text.", "high")
        risk += 30

    if employee_name and not name_appears_in_text(employee_name, raw_text):
        push_flag(flags, "employee_name_not_found", "Employee name was not confidently found in the resume.", "high")
        risk += 20

    if email and email.lower() not in raw_text.lower():
        push_flag(flags, "email_not_found", "Employee email was not found in the resume text.", "medium")
        risk += 10

    digits_contact = re.sub(r"\D+", "", contact_no or "")
    raw_digits = re.sub(r"\D+", "", raw_text)
    if digits_contact and digits_contact[-10:] not in raw_digits:
        push_flag(flags, "contact_number_not_found", "Employee contact number was not found in the resume text.", "medium")
        risk += 10

    if not features["has_work_experience"]:
        push_flag(flags, "missing_work_experience", "Resume does not clearly contain a work experience section.", "medium")
        risk += 10

    if not features["has_education"]:
        push_flag(flags, "missing_education", "Resume does not clearly contain an education section.", "medium")
        risk += 10

    if not features["has_skills"]:
        push_flag(flags, "missing_skills", "Resume does not clearly contain a skills section.", "low")
        risk += 5

    buzzwords = [
        "dynamic", "hardworking", "motivated", "results-driven",
        "team player", "detail-oriented", "fast learner", "highly organized"
    ]
    buzz_count = sum(1 for word in buzzwords if word in text)
    if buzz_count >= 4:
        push_flag(flags, "generic_resume_language", "Resume contains highly generic template-like language.", "medium")
        risk += 12

    if len(features["date_candidates"]) == 0:
        push_flag(flags, "missing_dates", "No clear date patterns were detected in the resume.", "medium")
        risk += 10

    if readability_data.get("readable") is False:
        push_flag(flags, "low_readability", f"Document readability issue: {readability_data.get('quality_reason', 'unknown')}.", "high")
        risk += 20

    risk = min(100, risk)
    risk_level, recommendation, is_suspicious = finalize_risk(risk)

    extracted_fields = {
        "employee_name": employee_name if employee_name else None,
        "email": email if email else None,
        "contact_no": contact_no if contact_no else None,
        **features
    }

    return {
        "risk_score": risk,
        "risk_level": risk_level,
        "is_suspicious": is_suspicious,
        "recommendation": recommendation,
        "flags": flags,
        "extracted_fields": extracted_fields
    }

def score_medical_clearance_risk(raw_text, employee_context, detected_type, readability_data):
    text = normalize_text(raw_text)
    flags = []
    risk = 0

    employee_name = (employee_context or {}).get("employee_name", "")
    fields = extract_medical_clearance_fields(raw_text)

    if detected_type not in ["medical_clearance", "others"]:
        push_flag(flags, "type_mismatch", f"Detected type is '{detected_type}' instead of 'medical_clearance'.", "high")
        risk += 35

    if len(text) < 80:
        push_flag(flags, "too_little_text", "Medical clearance contains too little readable text.", "high")
        risk += 30

    medical_keywords = [
        "medical clearance", "medical certificate", "fit to work",
        "medically fit", "physician", "clinic", "doctor"
    ]
    keyword_hits = sum(1 for k in medical_keywords if k in text)
    if keyword_hits < 2:
        push_flag(flags, "too_few_medical_keywords", "Too few medical-clearance keywords were found.", "high")
        risk += 25

    employee_name_found = name_appears_in_text(employee_name, raw_text) if employee_name else False
    fields["employee_name_in_text"] = employee_name_found

    if employee_name and not employee_name_found:
        push_flag(flags, "employee_name_not_found", "Employee name was not confidently found in the medical clearance.", "high")
        risk += 20

    if not fields["doctor_name"]:
        push_flag(flags, "doctor_name_missing", "Doctor name was not confidently detected.", "medium")
        risk += 12

    if not fields["clinic_name"]:
        push_flag(flags, "clinic_name_missing", "Clinic or hospital name was not confidently detected.", "medium")
        risk += 10

    if not fields["license_number"]:
        push_flag(flags, "license_number_missing", "License/PRC number was not confidently detected.", "medium")
        risk += 12

    if len(fields["date_candidates"]) == 0:
        push_flag(flags, "issue_date_missing", "No clear date was detected.", "medium")
        risk += 10

    if "fit to work" not in text and "medically fit" not in text:
        push_flag(flags, "fitness_phrase_missing", "Expected fitness wording was not detected.", "medium")
        risk += 10

    if readability_data.get("readable") is False:
        push_flag(flags, "low_readability", f"Document readability issue: {readability_data.get('quality_reason', 'unknown')}.", "high")
        risk += 20

    risk = min(100, risk)
    risk_level, recommendation, is_suspicious = finalize_risk(risk)

    return {
        "risk_score": risk,
        "risk_level": risk_level,
        "is_suspicious": is_suspicious,
        "recommendation": recommendation,
        "flags": flags,
        "extracted_fields": fields
    }

def score_generic_document_risk(raw_text, employee_context, expected_type, detected_type, readability_data):
    text = normalize_text(raw_text)
    flags = []
    risk = 0

    if len(text) < 50:
        push_flag(flags, "too_little_text", "Document contains too little readable text.", "high")
        risk += 30

    if expected_type and detected_type and expected_type != detected_type and detected_type != "others":
        push_flag(flags, "type_mismatch", f"Detected type is '{detected_type}' instead of '{expected_type}'.", "high")
        risk += 35

    employee_name = (employee_context or {}).get("employee_name", "")
    if employee_name and not name_appears_in_text(employee_name, raw_text):
        push_flag(flags, "employee_name_not_found", "Employee name was not confidently found in the document.", "medium")
        risk += 12

    if readability_data.get("readable") is False:
        push_flag(flags, "low_readability", f"Document readability issue: {readability_data.get('quality_reason', 'unknown')}.", "high")
        risk += 20

    risk = min(100, risk)
    risk_level, recommendation, is_suspicious = finalize_risk(risk)

    return {
        "risk_score": risk,
        "risk_level": risk_level,
        "is_suspicious": is_suspicious,
        "recommendation": recommendation,
        "flags": flags,
        "extracted_fields": {
            "employee_name": employee_name if employee_name else None
        }
    }

def finalize_risk(score):
    if score >= 80:
        return "critical", "reject", 1
    if score >= 60:
        return "high", "manual_review", 1
    if score >= 30:
        return "medium", "manual_review", 1
    return "low", "approve", 0

def verify_document_logic(file_path, ext, expected_type, employee_context):
    readability_data = {
        "readable": True,
        "ocr_confidence": None,
        "text_length": None,
        "quality_reason": "not_image"
    }

    if ext in [".jpg", ".jpeg", ".png"]:
        readability_data = check_image_readability(file_path)

    raw_text = extract_text(file_path, ext)
    detected_type, confidence = classify_by_content(raw_text) if raw_text.strip() else ("others", 0.50)

    if expected_type == "resume":
        result = score_resume_risk(raw_text, employee_context, detected_type, readability_data)
    elif expected_type == "medical_clearance":
        result = score_medical_clearance_risk(raw_text, employee_context, detected_type, readability_data)
    else:
        result = score_generic_document_risk(raw_text, employee_context, expected_type, detected_type, readability_data)

    return {
        "success": True,
        "document_type": detected_type,
        "confidence": round(confidence, 2),
        "expected_type": expected_type,
        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "is_suspicious": result["is_suspicious"],
        "recommendation": result["recommendation"],
        "flags": result["flags"],
        "extracted_fields": result["extracted_fields"],
        **readability_data
    }

# ==================================================
# API ENDPOINTS
# ==================================================
@app.route("/classify", methods=["POST"])
def classify():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file"}), 400

    ext = os.path.splitext(file.filename)[1].lower()

    allowed_exts = [".pdf", ".docx", ".jpg", ".jpeg", ".png", ".txt"]
    if ext not in allowed_exts:
        return jsonify({
            "document_type": "others",
            "confidence": 0.50,
            "readable": False,
            "ocr_confidence": None,
            "text_length": 0,
            "quality_reason": "unsupported_file_type"
        }), 400

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp:
            file.save(temp.name)
            temp_path = temp.name

        readability_data = {
            "readable": True,
            "ocr_confidence": None,
            "text_length": None,
            "quality_reason": "not_image"
        }

        if ext in [".jpg", ".jpeg", ".png"]:
            readability_data = check_image_readability(temp_path)

        text = extract_text(temp_path, ext)

        if not text.strip():
            return jsonify({
                "document_type": "others",
                "confidence": 0.50,
                **readability_data
            }), 200

        doc_type, confidence = classify_by_content(text)

        return jsonify({
            "document_type": doc_type,
            "confidence": round(confidence, 2),
            **readability_data
        }), 200

    except Exception as e:
        return jsonify({
            "document_type": "others",
            "confidence": 0.50,
            "readable": False,
            "ocr_confidence": None,
            "text_length": 0,
            "quality_reason": f"processing_error: {str(e)}"
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.route("/verify", methods=["POST"])
def verify():
    file = request.files.get("file")
    if not file:
        return jsonify({"success": False, "error": "No file"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    allowed_exts = [".pdf", ".docx", ".jpg", ".jpeg", ".png", ".txt"]
    if ext not in allowed_exts:
        return jsonify({
            "success": False,
            "error": "Unsupported file type",
            "document_type": "others",
            "confidence": 0.50,
            "risk_score": 100,
            "risk_level": "critical",
            "is_suspicious": 1,
            "recommendation": "reject",
            "flags": [
                {
                    "code": "unsupported_file_type",
                    "message": "Unsupported file type for verification.",
                    "severity": "high"
                }
            ],
            "extracted_fields": {}
        }), 400

    expected_type = (request.form.get("expected_type") or "").strip()
    employee_context = parse_request_json_field("employee_context", default={}) or {}

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp:
            file.save(temp.name)
            temp_path = temp.name

        result = verify_document_logic(
            file_path=temp_path,
            ext=ext,
            expected_type=expected_type,
            employee_context=employee_context
        )
        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "document_type": "others",
            "confidence": 0.50,
            "risk_score": 100,
            "risk_level": "critical",
            "is_suspicious": 1,
            "recommendation": "manual_review",
            "flags": [
                {
                    "code": "verification_processing_error",
                    "message": str(e),
                    "severity": "high"
                }
            ],
            "extracted_fields": {}
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

# ==================================================
# APP START
# ==================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
