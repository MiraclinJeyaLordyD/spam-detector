from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import numpy as np
import re
import math
from urllib.parse import urlparse

# ============================================
# Load trained model & dataset template
# ============================================
model = joblib.load("spam_model.pkl")

# Load dataset to extract EXACT feature structure
df_full = pd.read_csv("cleaned_dataset_final.csv")
df_full["label"] = df_full["label"].astype(int)
X_full = df_full.drop(columns=["label"])

# Template row EXACTLY like training
TEMPLATE_ROW = X_full.head(1).copy()
TEMPLATE_ROW.iloc[0] = 0

X_COLUMNS = X_full.columns.tolist()

# ============================================
# FastAPI setup
# ============================================
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================
# Build live features EXACT MATCH to training
# ============================================
def build_live_features(text, url):

    # Copy template row
    live = TEMPLATE_ROW.copy()
    live.iloc[0] = 0  # reset all features

    # Insert user inputs
    live.loc[live.index[0], "text"] = text
    live.loc[live.index[0], "URL"] = url

    # =============================
    # Recompute ONLY URL features (same as training code)
    # =============================
    u = str(url)

    live.loc[live.index[0], "URLLength"] = len(u)
    live.loc[live.index[0], "num_digits_url"] = sum(c.isdigit() for c in u)
    live.loc[live.index[0], "NoOfLettersInURL"] = sum(c.isalpha() for c in u)

    live.loc[live.index[0], "is_https"] = int(u.startswith("https://"))
    live.loc[live.index[0], "have_at"] = int("@" in u)
    live.loc[live.index[0], "have_ip"] = int(bool(re.match(r"http[s]?://\d+\.\d+\.\d+\.\d+", u)))

    live.loc[live.index[0], "NoOfQMarkInURL"] = u.count("?")
    live.loc[live.index[0], "NoOfAmpersandInURL"] = u.count("&")
    live.loc[live.index[0], "NoOfEqualsInURL"] = u.count("=")
    live.loc[live.index[0], "NoOfOtherSpecialCharsInURL"] = sum(c in "@#$%^*()" for c in u)

    live.loc[live.index[0], "url_num_dots"] = u.count(".")
    live.loc[live.index[0], "url_hyphen_count"] = u.count("-")
    live.loc[live.index[0], "url_underscore_count"] = u.count("_")

    dom = urlparse(u).netloc
    live.loc[live.index[0], "domain"] = dom
    live.loc[live.index[0], "tld"] = dom.split(".")[-1] if "." in dom else ""
    live.loc[live.index[0], "domain_length"] = len(dom)
    live.loc[live.index[0], "num_subdomain"] = dom.count(".")
    live.loc[live.index[0], "url_depth"] = urlparse(u).path.count("/")

    # Clean NaNs
    live.fillna(0, inplace=True)

    # EXACT training column order
    live = live[X_COLUMNS]

    return live

# ============================================
# Routes
# ============================================
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, text: str = Form(""), url: str = Form("")):

    df_live = build_live_features(text, url)

    proba = model.predict_proba(df_live)[0]
    ham = round(proba[0] * 100, 2)
    spam = round(proba[1] * 100, 2)

    # SAME threshold as training: 80%
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
