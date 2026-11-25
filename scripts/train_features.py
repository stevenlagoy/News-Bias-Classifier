"""Train a classifier on article features and show which feature groups carry information.

  python scripts/train_features.py data/articles/features.jsonl

Rows need "label" and "split" ("train" / "test", optionally "valid"). With no split values it falls back to
a publisher-disjoint 80/20 split. Everything is scored with macro-F1, the same metric as the text baselines.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # run from anywhere without installing

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from partisan_classifier.extraction.features import FEATURE_NAMES, load_features

N = FEATURE_NAMES
GROUPS = {
    "presentation": [f for f in N if f.startswith(("voice_", "quoted_slant", "quoted_stance", "lean_mean"))],
    "sources": [f for f in N if f.startswith(("quoted_lean", "quoted_left", "quoted_right", "src_", "n_src_"))],
    "claims": [f for f in N if f in {"n_claims", "log_n_claims", "attributed_share", "first_person_share"}
               or f.startswith(("opinion_", "personal_", "expert_", "speaker_"))],
    "coverage": ["n_primary", "n_primary_lean", "lean_coverage"],
}


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def macro_f1_ci(y, pred, rng, n=300):
    scores = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        scores.append(f1_score(y[i], pred[i], average="macro"))
    return np.percentile(scores, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ids, X, meta = load_features(args.features)
    y = np.array([m["label"] for m in meta])
    split = np.array([m.get("split") or "" for m in meta])
    pub = np.array([m.get("publisher") or "" for m in meta])
    if not (split == "train").any():
        tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=args.seed).split(X, y, pub))
        split = np.array(["train"] * len(y), dtype=object)
        split[te] = "test"
        print("no split field: using a publisher-disjoint 80/20 split", file=sys.stderr)
    rng = np.random.default_rng(args.seed)
    col = {f: i for i, f in enumerate(N)}
    masks = {s: split == s for s in ("train", "valid", "test") if (split == s).any()}

    print("articles per split/label:", {s: dict(Counter(map(str, y[m]))) for s, m in masks.items()})
    cov = X[:, col["lean_coverage"]]
    print(f"lean coverage: mean {cov.mean():.2f}; {np.mean(X[:, col['n_primary_lean']] == 0):.0%} of articles have "
          f"no primary subject with a known lean; {np.mean(X[:, col['quoted_lean_known_share']] == 0):.0%} have no "
          f"quoted source with a known lean\n")

    def run(model, cols, weighted=False):
        idx = [col[f] for f in cols]
        Xtr, ytr = X[masks["train"]][:, idx], y[masks["train"]]
        if weighted:   # estimators without class_weight get balanced sample weights instead
            model.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        else:
            model.fit(Xtr, ytr)
        return {s: model.predict(X[m][:, idx]) for s, m in masks.items() if s != "train"}

    rows = []
    def record(name, preds):
        rows.append((name, {s: f1_score(y[masks[s]], p, average="macro") for s, p in preds.items()}))

    record("majority class", run(DummyClassifier(strategy="most_frequent"), N))
    for g, cols in GROUPS.items():
        record(f"{g} only", run(logreg(), cols))
    for g in GROUPS:
        record(f"all minus {g}", run(logreg(), [f for f in N if f not in GROUPS[g]]))
    full = run(logreg(), N)
    record("all features, logistic", full)
    record("all features, gradient boosting", run(HistGradientBoostingClassifier(random_state=args.seed), N, weighted=True))

    splits = [s for s in ("valid", "test") if s in masks]
    print(f"{'macro-F1':36}" + "".join(f"{s:>9}" for s in splits))
    for name, sc in rows:
        print(f"{name:36}" + "".join(f"{sc.get(s, float('nan')):9.3f}" for s in splits))

    if "test" in masks:
        yt, pt = y[masks["test"]], full["test"]
        lo, hi = macro_f1_ci(yt, pt, rng)
        print(f"\nall features, logistic, test: macro-F1 {f1_score(yt, pt, average='macro'):.3f} "
              f"(95% bootstrap CI {lo:.3f} to {hi:.3f}), accuracy {accuracy_score(yt, pt):.3f}")
        print(classification_report(yt, pt, zero_division=0))
        labels = sorted(set(map(str, y)))
        print("confusion (rows = true, columns = predicted, order", labels, ")")
        print(confusion_matrix(yt, pt, labels=labels))

    model = logreg()
    model.fit(X[masks["train"]], y[masks["train"]])
    clf = model.named_steps["logisticregression"]
    print("\nlargest coefficients per class (standardized features):")
    for k, label in enumerate(clf.classes_):
        top = np.argsort(-np.abs(clf.coef_[k]))[:6]
        print(f"  {label:7}", ", ".join(f"{N[i]} {clf.coef_[k][i]:+.2f}" for i in top))


if __name__ == "__main__":
    main()