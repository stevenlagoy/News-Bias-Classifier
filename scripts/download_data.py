#!/usr/bin/env python
"""
CLI wrapper: download raw datasets into data/raw/.

Usage:
    python scripts/download_data.py
"""

from pathlib import Path

from partisan_classifier.data.download import (
    download_article_bias_prediction,
    download_qbias,
)

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    download_article_bias_prediction(raw_dir)
    download_qbias(raw_dir)
