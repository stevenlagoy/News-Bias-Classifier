"""Do the structured features add anything to a text model, on publishers the text model has never seen?

  python scripts/stack_features.py --src data/article-bias-prediction/data/jsons \
      --train-ids data/article-bias-prediction/data/splits/media/train.tsv \
      --features data/articles/features_v5.jsonl

Text model: TF-IDF + logistic regression on the official media-split train articles. Its predictions for the
feature-set articles in the train split are made by models that never saw that article's publisher (5-fold,
grouped by publisher), so they look like predictions on new publishers, which is what the test articles get.
Then three models are compared on the feature-set test articles: text only, features only, and both together.
The difference between 'both' and 'text only' comes with a publisher-resampled 95% interval.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from partisan_classifier.extraction.features import FEATURE_NAMES, load_features

LABELS = {0: "left", 1: "center", 2: "right"}
CLASSES = ["center", "left", "right"]


def norm_pub(name: str) -> str:
    """Publisher names compared loosely: 'The New York Times' and 'new york times' share a key."""
    return re.sub(r"^the", "", re.sub(r"[^a-z0-9]", "", str(name).lower()))


def read_ids(path: str) -> set[str]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    first = Path(path).read_text(encoding="utf-8").splitlines()[0]
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t" if "\t" in first else ",", quoting=csv.QUOTE_NONE))
    header = [c.strip().lower() for c in rows[0]]
    if "id" in header:
        col = header.index("id")
        return {r[col].strip() for r in rows[1:] if len(r) > col}
    return {r[0].strip() for r in rows if r}


def load_articles(src: str, ids: set[str], field: str) -> dict[str, dict]:
    out = {}
    for p in Path(src).glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        i = str(d.get("ID") or p.stem)
        if i in ids and str(d.get("bias", "")).strip() in {"0", "1", "2"}:
            text = d.get(field) or d.get("content") or d.get("content_original") or ""
            if text:
                out[i] = {"text": text, "label": LABELS[int(d["bias"])], "publisher": d.get("source") or "unknown"}
    return out


def cluster_ci(diffs: np.ndarray, groups: np.ndarray, rng, n: int = 400):
    """Publisher-resampled interval for a mean of per-article values."""
    members = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    keys = list(members)
    return np.percentile([diffs[np.concatenate([members[k] for k in rng.choice(keys, len(keys))])].mean()
                          for _ in range(n)], [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder of Article-Bias-Prediction .json files")
    ap.add_argument("--train-ids", required=True, help="the split file listing the train articles")
    ap.add_argument("--features", required=True, help="features JSONL with split = train/test")
    ap.add_argument("--text-field", default="content", help="json field for the text model (content or content_original)")
    ap.add_argument("--max-features", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    ids, X, meta = load_features(args.features)
    y = np.array([m["label"] for m in meta])
    split = np.array([m.get("split") or "" for m in meta])
    pub = np.array([m.get("publisher") or "" for m in meta])
    tr_sub, te_sub = np.where(split == "train")[0], np.where(split == "test")[0]
    print(f"feature set: {len(tr_sub)} train and {len(te_sub)} test articles", file=sys.stderr)

    train_ids = read_ids(args.train_ids)
    arts = load_articles(args.src, train_ids, args.text_field)
    tid = sorted(arts)
    print(f"text model: {len(tid)} train articles, {len({arts[i]['publisher'] for i in tid})} publishers", file=sys.stderr)
    ttext = [arts[i]["text"] for i in tid]
    ty = np.array([arts[i]["label"] for i in tid])
    tpub = np.array([norm_pub(arts[i]["publisher"]) for i in tid])      # folds grouped by the loose name
    train_pubs = set(tpub)
    pos = {i: k for k, i in enumerate(tid)}

    vec = TfidfVectorizer(max_features=args.max_features, sublinear_tf=True, ngram_range=(1, 2), min_df=3, stop_words="english")
    T = vec.fit_transform(ttext)
    # test text for the feature-set test articles
    test_art = load_articles(args.src, {ids[i] for i in te_sub}, args.text_field)
    keep_te = np.array([i for i in te_sub if ids[i] in test_art])
    Tte = vec.transform([test_art[ids[i]]["text"] for i in keep_te])

    # text probabilities: train-subset articles from models that never saw their publisher; test from the fold average
    oof = {}
    te_prob = np.zeros((len(keep_te), 3))
    sub_in = [i for i in tr_sub if ids[i] in pos]
    sub_rows = np.array([pos[ids[i]] for i in sub_in])
    for k, (a, b) in enumerate(GroupKFold(5).split(T, ty, tpub)):
        m = LogisticRegression(max_iter=1000, C=10, class_weight="balanced").fit(T[a], ty[a])
        order = [list(m.classes_).index(c) for c in CLASSES]
        hit = np.intersect1d(b, sub_rows)
        if len(hit):
            p = m.predict_proba(T[hit])[:, order]
            for row, pr in zip(hit, p):
                oof[row] = pr
        te_prob += m.predict_proba(Tte)[:, order] / 5
        print(f"  text fold {k + 1}/5 done", file=sys.stderr)
    sub_ok = [(i, pos[ids[i]]) for i in sub_in if pos[ids[i]] in oof]
    tr_idx = np.array([i for i, _ in sub_ok])
    tr_prob = np.array([oof[r] for _, r in sub_ok])
    te_idx = keep_te

    def meta_model(Xtr, ytr, gtr):
        grid = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
                            {"logisticregression__C": [0.01, 0.1, 1.0]}, scoring="f1_macro",
                            cv=GroupKFold(5).split(Xtr, ytr, gtr))
        return grid.fit(Xtr, ytr)

    text_tr, text_te = np.log(tr_prob + 1e-6), np.log(te_prob + 1e-6)
    text_pred = np.array(CLASSES)[te_prob.argmax(1)]
    m_feat = meta_model(X[tr_idx], y[tr_idx], pub[tr_idx])
    feat_pred = m_feat.predict(X[te_idx])
    m_both = meta_model(np.hstack([text_tr, X[tr_idx]]), y[tr_idx], pub[tr_idx])
    both_pred = m_both.predict(np.hstack([text_te, X[te_idx]]))

    yt, gt = y[te_idx], pub[te_idx]
    seen = np.array([norm_pub(p) in train_pubs for p in gt])
    groups = {"all test articles": np.ones(len(yt), bool),
              "publisher ALSO in train (by name): text model has seen the outlet": seen,
              "publisher NOT in train: genuinely new outlets": ~seen}
    preds = {"text only (TF-IDF)": text_pred, "features only": feat_pred, "text + features": both_pred}
    for gname, mask in groups.items():
        if mask.sum() == 0:
            continue
        print(f"\n{gname}: {mask.sum()} articles from {len(set(gt[mask]))} publishers")
        print(f"  {'model':26}{'macro-F1':>10}{'accuracy':>10}")
        for name, pred in preds.items():
            print(f"  {name:26}{f1_score(yt[mask], pred[mask], average='macro'):10.3f}{np.mean(pred[mask] == yt[mask]):10.3f}")
        if len(set(gt[mask])) >= 5:
            for a_name, b_name in (("text + features", "text only (TF-IDF)"), ("features only", "text only (TF-IDF)")):
                d = (preds[a_name][mask] == yt[mask]).astype(float) - (preds[b_name][mask] == yt[mask]).astype(float)
                lo, hi = cluster_ci(d, gt[mask], rng)
                word = "better than" if lo > 0 else "worse than" if hi < 0 else "not distinguishable from"
                print(f"  {a_name} minus {b_name}, accuracy: {d.mean():+.3f} (95% interval {lo:+.3f} to {hi:+.3f}): {word}")
        else:
            print("  too few publishers for an interval")
    print("\nNote: the text model's score on 'publisher ALSO in train' is helped by having seen the outlet. "
          "Only the second group says how well each model does on outlets it has never seen.")
    print("\nlargest features in the combined model (standardized):")
    clf = m_both.best_estimator_.named_steps["logisticregression"]
    names = ["text:" + c for c in CLASSES] + FEATURE_NAMES
    for k, label in enumerate(clf.classes_):
        top = np.argsort(-np.abs(clf.coef_[k]))[:6]
        print(f"  {label:7}", ", ".join(f"{names[i]} {clf.coef_[k][i]:+.2f}" for i in top))


if __name__ == "__main__":
    main()