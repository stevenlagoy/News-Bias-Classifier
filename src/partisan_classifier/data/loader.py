"""
Load raw/interim datasets into a common in-memory representation.

The target shape downstream code should expect is one row per article with,
at minimum: article text, publisher/outlet, and a left/center/right label.
Keeping this schema stable here is what lets the rest of the pipeline stay
agnostic to which raw dataset it came from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["text", "publisher", "label"]

_BIAS_MAP = {0: "left", 1: "center", 2: "right"}

_OOD_SECONDARY_BIAS_MAP = {
    "left": "left",
    "leaning-left": "left",
    "center": "center",
    "leaning-right": "right",
    "right": "right",
}


def load_article_bias_prediction(raw_dir: str | Path) -> pd.DataFrame:
    """Load the Article-Bias-Prediction dataset into REQUIRED_COLUMNS schema."""
    base = Path(raw_dir) / "article-bias-prediction" / "data" / "jsons"
    rows = []
    for f in base.glob("*.json"):
        with open(f) as fh:
            article = json.load(fh)
        bias = article.get("bias_text") or _BIAS_MAP.get(article.get("bias"))
        rows.append({
            "text": article.get("content", ""),
            "publisher": article.get("source", ""),
            "label": bias,
            "article_id": article.get("ID"),
        })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip() != ""]
    validate_schema(df)
    return df

def load_qbias(raw_dir: str | Path) -> pd.DataFrame:
    """Load Qbias into REQUIRED_COLUMNS schema for OOD evaluation."""
    path = Path(raw_dir) / "qbias" / "allsides_balanced_news_headlines-texts.csv"
    raw = pd.read_csv(path)
    print("Qbias columns: ", list(raw.columns)) # confirm names
    df = pd.DataFrame({
        "text": raw.get("text", raw.get("heading", "")),
        "publisher": raw.get("source", raw.get("news_source", "")),
        "label": raw.get("bias_rating", raw.get("bias", "")).str.lower(),
    })
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip() != ""]
    validate_schema(df)
    return df


def load_ood_secondary(raw_dir: str | Path) -> pd.DataFrame:
    """Load second OOD test set (Kaggle) into REQUIRED_COLUMNS schema."""
    path = Path(raw_dir) / "ood-secondary"
    csv_files = list(path.glob("*.csv"))
    raw = pd.read_csv(csv_files[0])
    df = pd.DataFrame({
        "text": raw.get("page_text", ""),
        "publisher": raw.get("site", ""),
        "label": raw.get("bias", "").astype(str).str.lower().map(_OOD_SECONDARY_BIAS_MAP),
    })
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip() != ""]
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
