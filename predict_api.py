from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import numpy as np
import re
import math
from urllib.parse import urlparse

# Load Model
model = joblib.load("spam_model.pkl")
model_features = list(model.feature_names_in_)

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

SHORTENERS = ["bit.ly","tinyurl","t.co","goo.gl","is.gd","cutt.ly",
              "rebrand.ly","shorturl","shorte","rb.gy"]

BAD_TLDS = ["xyz","top","loan","win","gift","click","shop","rest","live"]

def entropy(s):
    if not s:
        return 0
    probs = [s.count(c)/len(s) for c in dict.fromkeys(s)]
    return -sum(p * math.log(p, 2) for p in probs)

def detect_lang(text):
    if re.search(r"[\u0900-\u097F]", text): return "hi"
    if re.search(r"[\u0980-\u09FF]", text): return "bn"
    if re.search(r"[\u0B00-\u0B7F]", text): return "or"
    return "en"

def suspicious_score(text):
    score = 0
    text_low = text.lower()
    if any(k in text_low for k in ["win","free","gift","offer","click","urgent"]):
        score += 1
    if re.search(r"\b\d{10}\b", text_low):
        score += 1
    if "http" in text_low:
        score += 1
    return score

def build_live_features(text, url):
    df = pd.DataFrame({col: [0] for col in model_features})
    df["text"] = text
    df["URL"] = url

    df["text_num_chars"] = len(text)
    df["text_num_words"] = len(text.split())
    df["text_num_digits"] = sum(c.isdigit() for c in text)
    df["text_avg_word_len"] = np.mean([len(w) for w in text.split()]) if text.split() else 0
    df["num_emoji"] = sum(ord(c) > 10000 for c in text)
    df["text_upper_ratio"] = sum(c.isupper() for c in text) / len(text) if len(text) else 0
    df["text_punct_ratio"] = sum(c in "!@#$%^&*?/\\|" for c in text) / len(text) if len(text) else 0
    df["text_contains_phone"] = int(bool(re.search(r"\b\d{10}\b", text)))
    df["text_contains_email"] = int(bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)))
    df["text_contains_shortlink"] = int(any(s in text.lower() for s in SHORTENERS))
    df["text_suspicious_score"] = suspicious_score(text)
    df["lang_detected"] = detect_lang(text)

    domain = urlparse(url).netloc
    tld = domain.split(".")[-1] if "." in domain else ""

    df["URLLength"] = len(url)
    df["num_digits_url"] = sum(c.isdigit() for c in url)
    df["NoOfLettersInURL"] = sum(c.isalpha() for c in url)
    df["is_https"] = int(url.startswith("https://"))
    df["have_at"] = int("@" in url)
    df["have_ip"] = int(bool(re.match(r"http[s]?://\d+\.\d+\.\d+\.\d+", url)))
    df["NoOfQMarkInURL"] = url.count("?")
    df["NoOfAmpersandInURL"] = url.count("&")
    df["NoOfEqualsInURL"] = url.count("=")
    df["NoOfOtherSpecialCharsInURL"] = sum(c in "@#$%^*()" for c in url)
    df["url_num_dots"] = url.count(".")
    df["url_hyphen_count"] = url.count("-")
    df["url_underscore_count"] = url.count("_")
    df["tinyurl"] = int(any(s in url.lower() for s in SHORTENERS))

    df["domain"] = domain
    df["tld"] = tld
    df["domain_length"] = len(domain)
    df["num_subdomain"] = domain.count(".")
    df["url_depth"] = urlparse(url).path.count("/")
    df["url_bad_tld"] = int(tld in BAD_TLDS)
    df["url_entropy"] = entropy(url)

    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    return df[model_features]

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, text: str = Form(""), url: str = Form("")):
    df_live = build_live_features(text, url)
    proba = model.predict_proba(df_live)[0]
    ham = round(100 * proba[0], 2)
    spam = round(100 * proba[1], 2)

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
