import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from partisan_classifier.data.loader import (
    load_article_bias_prediction,
    load_media_split,
    load_ood_secondary,
    load_qbias,
)
from partisan_classifier.evaluation.metrics import compute_metrics
from partisan_classifier.models.baseline import MajorityClassBaseline, TfidfLogisticBaseline

RAW_DIR = "data/raw"
LABELS = ["left", "center", "right"]


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def describe(name: str, df: pd.DataFrame) -> dict:
    lengths = df["text"].str.len()
    info = {
        "n": len(df),
        "labels": df["label"].value_counts().reindex(LABELS, fill_value=0).to_dict(),
        "publishers": int(df["publisher"].nunique()),
        "publishers_per_label": (
            df.groupby("label")["publisher"].nunique().reindex(LABELS, fill_value=0).to_dict()
        ),
        "len_mean": round(float(lengths.mean())),
        "len_median": round(float(lengths.median())),
        "len_min": int(lengths.min()),
        "len_max": int(lengths.max()),
    }
    print(f"[{name}] {info}")
    return info


def summarize(model_name: str, set_name: str, y_true, y_pred, results: dict) -> None:
    m = compute_metrics(y_true, y_pred)
    per = m["per_class_report"]
    f1 = {l: per[l]["f1-score"] for l in LABELS}
    rec = {l: per[l]["recall"] for l in LABELS}
    results[f"{model_name}|{set_name}"] = {
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "f1": f1,
        "recall": rec,
        "confusion_matrix": m["confusion_matrix"],
    }
    print(
        f"{model_name:<20} {set_name:<13} acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
        f"F1(L/C/R)={f1['left']:.2f}/{f1['center']:.2f}/{f1['right']:.2f} "
        f"recall(L/C/R)={rec['left']:.2f}/{rec['center']:.2f}/{rec['right']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-bert", action="store_true",
                        help="also run politicalBiasBERT on the official test split")
    args = parser.parse_args()

    report: dict = {"datasets": {}, "results": {}}

    full_df = load_article_bias_prediction(RAW_DIR)
    train_df, val_df, test_df = load_media_split(RAW_DIR)
    qbias_df = load_qbias(RAW_DIR)
    kaggle_df = load_ood_secondary(RAW_DIR)

    print("=== DATASETS ===")
    for name, d in [("full", full_df), ("media_train", train_df), ("media_val", val_df),
                    ("media_test", test_df), ("qbias", qbias_df), ("kaggle", kaggle_df)]:
        report["datasets"][name] = describe(name, d)

    print("\n=== OFFICIAL SPLIT INTEGRITY ===")
    pubs = {"train": set(train_df["publisher"]), "val": set(val_df["publisher"]),
            "test": set(test_df["publisher"])}
    overlap = {
        "train&val": len(pubs["train"] & pubs["val"]),
        "train&test": len(pubs["train"] & pubs["test"]),
        "val&test": len(pubs["val"] & pubs["test"]),
    }
    print(f"[publisher overlap between splits] {overlap}")
    used_ids = set(train_df["article_id"]) | set(val_df["article_id"]) | set(test_df["article_id"])
    excluded = full_df[~full_df["article_id"].isin(used_ids)]
    print(f"[articles in full set but in no official split] n={len(excluded)} "
          f"labels={excluded['label'].value_counts().to_dict()} "
          f"publishers={excluded['publisher'].nunique()}")
    report["split_integrity"] = {"publisher_overlap": overlap, "excluded_n": len(excluded)}

    print("\n=== HOW OUT-OF-DISTRIBUTION ARE THE OOD SETS? (publisher names, approximate) ===")
    train_pubs = {_norm(p) for p in train_df["publisher"]}
    for name, d in [("qbias", qbias_df), ("kaggle", kaggle_df)]:
        share = d["publisher"].map(_norm).isin(train_pubs).mean()
        print(f"[{name}] share of articles whose publisher also appears in official train: {share:.1%}")
        print(f"[{name}] top publishers: {d['publisher'].value_counts().head(12).to_dict()}")

    print("\n=== MODELS ON OFFICIAL MEDIA SPLIT ===")
    eval_sets = {"media_val": val_df, "media_test": test_df, "qbias": qbias_df, "kaggle": kaggle_df}

    majority = MajorityClassBaseline().fit([], train_df["label"].tolist())
    print(f"[majority label from official train] {majority.majority_label}")
    tfidf = TfidfLogisticBaseline().fit(train_df["text"].tolist(), train_df["label"].tolist())

    for model_name, model in [("majority", majority), ("tfidf_logreg", tfidf)]:
        for set_name, d in eval_sets.items():
            preds = model.predict(d["text"].tolist())
            summarize(model_name, set_name, d["label"].tolist(), preds, report["results"])

    if args.with_bert:
        from partisan_classifier.models.pretrained_baseline import PoliticalBiasBertBaseline

        bert = PoliticalBiasBertBaseline()
        preds = bert.predict(test_df["text"].tolist())
        summarize("politicalBiasBERT", "media_test", test_df["label"].tolist(), preds, report["results"])

    print("\n=== RANDOM-SPLIT CONTROL (same TF-IDF model, article-level 80/20) ===")
    r_train, r_val = train_test_split(
        full_df, test_size=0.2, random_state=42, stratify=full_df["label"]
    )
    control = TfidfLogisticBaseline().fit(r_train["text"].tolist(), r_train["label"].tolist())
    preds = control.predict(r_val["text"].tolist())
    summarize("tfidf_logreg", "random_split", r_val["label"].tolist(), preds, report["results"])

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    (out / "baseline_report.json").write_text(json.dumps(report, indent=2, default=_json_default))
    print("\nSaved outputs/baseline_report.json")


if __name__ == "__main__":
    main()
