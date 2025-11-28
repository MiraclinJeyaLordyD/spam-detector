from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse

# Load the LR model pipeline
model = joblib.load("spam_model.pkl")
model_features = list(model.feature_names_in_)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# === SAME FUNCTION USED IN TRAINING ===
def compute_live_url_features(df_row):
    url = str(df_row.loc[df_row.index[0], "URL"])

    df_row.loc[df_row.index[0], "URLLength"] = len(url)
    df_row.loc[df_row.index[0], "num_digits_url"] = sum(c.isdigit() for c in url)
    df_row.loc[df_row.index[0], "NoOfLettersInURL"] = sum(c.isalpha() for c in url)

    df_row.loc[df_row.index[0], "is_https"] = int(url.startswith("https://"))
    df_row.loc[df_row.index[0], "have_at"] = int("@" in url)
    df_row.loc[df_row.index[0], "have_ip"] = int(bool(re.match(r"http[s]?://\d+\.\d+\.\d+\.\d+", url)))

    df_row.loc[df_row.index[0], "NoOfQMarkInURL"] = url.count("?")
    df_row.loc[df_row.index[0], "NoOfAmpersandInURL"] = url.count("&")
    df_row.loc[df_row.index[0], "NoOfEqualsInURL"] = url.count("=")
    df_row.loc[df_row.index[0], "NoOfOtherSpecialCharsInURL"] = sum(c in "@#$%^*()" for c in url)

    df_row.loc[df_row.index[0], "url_num_dots"] = url.count(".")
    df_row.loc[df_row.index[0], "url_hyphen_count"] = url.count("-")
    df_row.loc[df_row.index[0], "url_underscore_count"] = url.count("_")

    dom = urlparse(url).netloc
    df_row.loc[df_row.index[0], "domain"] = dom
    df_row.loc[df_row.index[0], "tld"] = dom.split(".")[-1] if "." in dom else ""
    df_row.loc[df_row.index[0], "domain_length"] = len(dom)
    df_row.loc[df_row.index[0], "num_subdomain"] = dom.count(".")
    df_row.loc[df_row.index[0], "url_depth"] = urlparse(url).path.count("/")

    return df_row


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(request: Request, text: str = Form(""), url: str = Form("")):

    # Create row identical to training dataset
    df_live = pd.DataFrame({col: [0] for col in model_features})

    df_live.loc[df_live.index[0], "text"] = text
    df_live.loc[df_live.index[0], "URL"] = url

    # Compute same URL features used during training
    df_live = compute_live_url_features(df_live)

    # Predict
    proba = model.predict_proba(df_live)[0]
    ham = round(proba[0] * 100, 2)
    spam = round(proba[1] * 100, 2)

    # SPAM threshold 80%
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

