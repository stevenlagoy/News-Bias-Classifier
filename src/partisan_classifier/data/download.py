"""
Download raw datasets into data/raw/.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import requests

ARTICLE_BIAS_REPO = "https://github.com/ramybaly/Article-Bias-Prediction.git"
QBIAS_CSV_URL = (
    "https://raw.githubusercontent.com/irgroup/Qbias/main/"
    "allsides_balanced_news_headlines-texts.csv"
)


def download_article_bias_prediction(dest_dir: str | Path) -> None:
    """Clone Article-Bias-Prediction into dest_dir/article-bias-prediction."""
    dest_dir = Path(dest_dir)
    target = dest_dir / "article-bias-prediction"
    if target.exists():
        return
    subprocess.run(
        ["git", "clone", "--depth", "1", ARTICLE_BIAS_REPO, str(target)],
        check=True,
    )


def download_qbias(dest_dir: str | Path) -> None:
    """Fetch the Qbias AllSides CSV into dest_dir/qbias/."""
    dest_dir = Path(dest_dir) / "qbias"
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / "allsides_balanced_news_headlines-texts.csv"
    if out_path.exists():
        return
    resp = requests.get(QBIAS_CSV_URL, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)

def download_ood_secondary(dest_dir: str | Path) -> None:
    """
    Second OOD test set (Kaggle). Requires Kaggle API credentials
    (~/.kaggle/kaggle.json) and `pip install kaggle`.
    """
    dest_dir = Path(dest_dir) / "ood-secondary"
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "gandpablo/news-articles-for-political-bias-classification",
            "-p", str(dest_dir), "--unzip"
        ],
        check=True
    )


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    download_article_bias_prediction(raw_dir)
    download_qbias(raw_dir)
    download_ood_secondary(raw_dir)
