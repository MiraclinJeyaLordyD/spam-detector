from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import numpy as np
import re
import math
from urllib.parse import urlparse

# ==============================
# Load Model
# ==============================
model = joblib.load("spam_model.pkl")

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==============================
# Utilities
# ==============================
def clean_text(t):
    if t is None:
        return ""
    t = str(t).strip()
    return re.sub(r"\s+", " ", t)

def detect_lang(text):
    if re.search(r"[\u0900-\u097F]", text): return "hi"
    if re.search(r"[\u0980-\u09FF]", text): return "bn"
    if re.search(r"[\u0B00-\u0B7F]", text): return "or"
    return "en"

def suspicious_score(text):
    score = 0
    text_low = text.lower()

    if any(k in text_low for k in ["win", "free", "gift", "offer", "click", "urgent"]):
        score += 1
    if re.search(r"\b\d{10}\b", text_low):
        score += 1
    if "http" in text_low:
        score += 1
    return score

SHORTENERS = ["bit.ly","tinyurl","t.co","goo.gl","is.gd","cutt.ly",
              "rebrand.ly","shorturl","shorte","rb.gy"]

BAD_TLDS = ["xyz","top","loan","win","gift","click","shop","rest","live"]

def entropy(s):
    if not s:
        return 0
    probs = [s.count(c)/len(s) for c in dict.fromkeys(s)]
    return -sum(p * math.log(p, 2) for p in probs)

# ==============================
# Feature Builder
# ==============================
def build_live_features(text, url):
    text = clean_text(text)
    url = clean_text(url)

    domain = urlparse(url).netloc
    tld = domain.split(".")[-1] if "." in domain else ""

    feats = {
        # TEXT FEATURES
        "text": text,
        "text_num_chars": len(text),
        "text_num_words": len(text.split()),
        "text_num_digits": sum(c.isdigit() for c in text),
        "text_avg_word_len": np.mean([len(w) for w in text.split()])
            if len(text.split()) else 0,
        "num_emoji": sum(ord(c) > 10000 for c in text),
        "text_upper_ratio": sum(c.isupper() for c in text) / len(text)
            if len(text) else 0,
        "text_punct_ratio": sum(c in "!@#$%^&*?/\\|" for c in text) / len(text)
            if len(text) else 0,
        "text_contains_phone": 1 if re.search(r"\b\d{10}\b", text) else 0,
        "text_contains_email": 1 if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text) else 0,
        "text_contains_shortlink": 1 if any(s in text.lower() for s in SHORTENERS) else 0,
        "text_suspicious_score": suspicious_score(text),
        "lang_detected": detect_lang(text),

        # URL STRING
        "URL": url,

        # URL FEATURES
        "URLLength": len(url),
        "num_digits_url": sum(c.isdigit() for c in url),
        "NoOfLettersInURL": sum(c.isalpha() for c in url),
        "is_https": 1 if url.startswith("https://") else 0,
        "have_at": 1 if "@" in url else 0,
        "have_ip": 1 if re.match(r"http[s]?://\d+\.\d+\.\d+\.\d+", url) else 0,
        "NoOfQMarkInURL": url.count("?"),
        "NoOfAmpersandInURL": url.count("&"),
        "NoOfEqualsInURL": url.count("="),
        "NoOfOtherSpecialCharsInURL": sum(c in "@#$%^*()" for c in url),
        "url_num_dots": url.count("."),
        "url_hyphen_count": url.count("-"),
        "url_underscore_count": url.count("_"),
        "tinyurl": 1 if any(s in url.lower() for s in SHORTENERS) else 0,

        # DOMAIN FEATURES
        "domain": domain,
        "tld": tld,
        "domain_length": len(domain),
        "num_subdomain": domain.count("."),
        "url_depth": urlparse(url).path.count("/"),
        "url_bad_tld": 1 if tld in BAD_TLDS else 0,
        "url_entropy": entropy(url),
    }

    return pd.DataFrame([feats])

# ==============================
# Routes
# ==============================
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, text: str = Form(""), url: str = Form("")):
    df_live = build_live_features(text, url)

    proba = model.predict_proba(df_live)[0]
    ham = round(proba[0] * 100, 2)
    spam = round(proba[1] * 100, 2)

    result = "SPAM 🚨" if spam >= 80 else "HAM ✅"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "ham": ham,
            "spam": spam,
            "text": text,
            "url": url,
        }
    )
