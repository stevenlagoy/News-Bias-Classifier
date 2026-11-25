"""Evaluate structured article features, including feature-group and ablation experiments.

Example:
    python scripts/train_features.py data/articles/features_v5.jsonl

Expected rows contain id, label, publisher, split ("train", "valid", "test"),
and a numeric "features" object. If no train/test split is present, a
publisher-disjoint 80/20 split is created.

All model selection uses training data only. Validation and test sets are never
used to fit scalers or select hyperparameters. Publisher-cluster bootstrap
intervals are reported for held-out metrics.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, GridSearchCV, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from partisan_classifier.extraction.features import FEATURE_NAMES, load_features


GROUPS = {
    "presentation": [
        f for f in FEATURE_NAMES
        if f.startswith(("voice_", "quoted_slant", "quoted_stance", "lean_mean"))
    ],
    "sources": [
        f for f in FEATURE_NAMES
        if f.startswith(("quoted_lean", "quoted_left", "quoted_right", "src_", "n_src_"))
    ],
    "claims": [
        f for f in FEATURE_NAMES
        if f in {
            "n_claims", "log_n_claims", "attributed_share", "first_person_share",
            "opinion_share_voice", "opinion_share_attr", "personal_opinion_share",
            "expert_opinion_share",
        }
        or f.startswith(("speaker_", "opinion_", "personal_", "expert_"))
    ],
    "coverage": ["n_primary", "n_primary_lean", "lean_coverage"],
}
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


def publisher_groups(meta: list[dict]) -> np.ndarray:
    return np.array([
        str(m.get("publisher") or "unknown").strip().casefold()
        for m in meta
    ])


def logistic():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    )


def grouped_cv(groups: np.ndarray):
    n = len(np.unique(groups))
    if n < 2:
        raise ValueError("At least two distinct publishers are required in the training split.")
    return list(GroupKFold(n_splits=min(5, n)).split(np.zeros(len(groups)), groups, groups))


def fit_logistic(Xtr, ytr, gtr):
    cv = grouped_cv(gtr)
    search = GridSearchCV(
        logistic(),
        {"logisticregression__C": C_GRID},
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(Xtr, ytr)
    return search


def cluster_bootstrap_ci(y, pred, groups, rng, n_boot=1000):
    """Cluster bootstrap of macro-F1 and accuracy, resampling publishers."""
    members = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    keys = np.array(list(members), dtype=object)
    f1s, accs = [], []
    for _ in range(n_boot):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([members[g] for g in sampled])
        f1s.append(f1_score(y[idx], pred[idx], average="macro", zero_division=0))
        accs.append(accuracy_score(y[idx], pred[idx]))
    return (
        np.percentile(f1s, [2.5, 97.5]),
        np.percentile(accs, [2.5, 97.5]),
    )


def metrics(y, pred):
    return f1_score(y, pred, average="macro", zero_division=0), accuracy_score(y, pred)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--no-individual-ablation", action="store_true",
                    help="skip the one-feature-at-a-time ablation table")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ids, X, meta = load_features(args.features)
    y = np.array([m["label"] for m in meta])
    split = np.array([str(m.get("split") or "").strip().lower() for m in meta])
    pub = publisher_groups(meta)

    if not np.any(split == "train") or not np.any(np.isin(split, ["valid", "test"])):
        tr, te = next(GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=args.seed
        ).split(X, y, groups=pub))
        split = np.full(len(y), "train", dtype=object)
        split[te] = "test"
        print("No usable train/valid/test split found: using publisher-disjoint 80/20 train/test.")

    masks = {s: split == s for s in ("train", "valid", "test") if np.any(split == s)}
    if "train" not in masks:
        raise ValueError("Feature file has no rows with split='train'.")

    train_idx = np.flatnonzero(masks["train"])
    ytr, gtr = y[train_idx], pub[train_idx]
    print(f"Rows: {len(y)}; publishers: {len(np.unique(pub))}")
    print("Split sizes:", {s: int(m.sum()) for s, m in masks.items()})
    print("Training labels:", dict(Counter(map(str, ytr))))
    print("Training publishers:", len(np.unique(gtr)))

    feature_index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    used = set()
    for name, cols in GROUPS.items():
        missing = [c for c in cols if c not in feature_index]
        if missing:
            raise ValueError(f"Unknown columns in group {name}: {missing}")
        overlap = used.intersection(cols)
        if overlap:
            raise ValueError(f"Features assigned to multiple groups: {sorted(overlap)}")
        used.update(cols)
    unassigned = [f for f in FEATURE_NAMES if f not in used]
    if unassigned:
        print("Features not assigned to a named group (still included in 'all'):", ", ".join(unassigned))

    # Each experiment selects C using publisher-grouped CV on train only.
    experiments: list[tuple[str, list[int]]] = [
        ("all features", list(range(X.shape[1]))),
    ]
    for group, cols in GROUPS.items():
        experiments.append((f"{group} only", [feature_index[c] for c in cols]))
    for group, cols in GROUPS.items():
        excluded = set(cols)
        experiments.append((
            f"all minus {group}",
            [i for i, name in enumerate(FEATURE_NAMES) if name not in excluded],
        ))

    results = {}
    print("\nFeature-group experiments (each logistic model tunes C using grouped CV):")
    print(f"{'model':28s}" + "".join(f"{s:>22s}" for s in masks if s != "train"))
    for name, cols in experiments:
        if not cols:
            continue
        model = fit_logistic(X[train_idx][:, cols], ytr, gtr)
        results[name] = model
        row = f"{name:28s}"
        for s, mask in masks.items():
            if s == "train":
                continue
            pred = model.predict(X[mask][:, cols])
            f1, acc = metrics(y[mask], pred)
            row += f"{f1:.3f}/{acc:.3f}".rjust(22)
        print(row + f"   C={model.best_params_['logisticregression__C']}")

    # A simple non-linear comparison, with balanced sample weights.
    hgb = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=args.seed,
    )
    hgb.fit(
        X[train_idx], ytr,
        sample_weight=compute_sample_weight("balanced", ytr),
    )
    print("\nAll features, gradient boosting:")
    for s, mask in masks.items():
        if s == "train":
            continue
        pred = hgb.predict(X[mask])
        f1, acc = metrics(y[mask], pred)
        print(f"  {s:5s}: macro-F1={f1:.3f}, accuracy={acc:.3f}")

    # Bootstrap intervals for held-out all-feature model.
    full = results["all features"]
    for s in ("valid", "test"):
        if s not in masks:
            continue
        mask = masks[s]
        pred = full.predict(X[mask])
        f1, acc = metrics(y[mask], pred)
        f1_ci, acc_ci = cluster_bootstrap_ci(
            y[mask], pred, pub[mask], rng, args.bootstrap
        )
        print(
            f"\nAll features, {s}: macro-F1={f1:.3f} "
            f"(publisher-bootstrap 95% CI {f1_ci[0]:.3f} to {f1_ci[1]:.3f}); "
            f"accuracy={acc:.3f} "
            f"(95% CI {acc_ci[0]:.3f} to {acc_ci[1]:.3f})"
        )

    # Individual leave-one-feature-out ablation. Fix C at the all-feature
    # training-CV choice so the comparison isolates feature removal.
    if not args.no_individual_ablation:
        best_c = full.best_params_["logisticregression__C"]
        print(f"\nIndividual feature ablation (fixed C={best_c}; change is ablated minus full):")
        print(f"{'removed feature':34s}" + "".join(f"{s:>22s}" for s in masks if s != "train"))
        for j, feature in enumerate(FEATURE_NAMES):
            cols = [i for i in range(X.shape[1]) if i != j]
            model = logistic().set_params(logisticregression__C=best_c)
            model.fit(X[train_idx][:, cols], ytr)
            row = f"{feature:34s}"
            for s, mask in masks.items():
                if s == "train":
                    continue
                pred_full = full.predict(X[mask])
                pred_ablated = model.predict(X[mask][:, cols])
                f1_full, acc_full = metrics(y[mask], pred_full)
                f1_ab, acc_ab = metrics(y[mask], pred_ablated)
                row += f"{(f1_ab-f1_full):+.3f}/{(acc_ab-acc_full):+.3f}".rjust(22)
            print(row)

    print("\nInterpretation: positive ablation deltas mean removing that feature improved the score;")
    print("negative deltas mean the feature helped. Treat small differences as noise, especially")
    print("when the held-out set contains relatively few independent publishers.")


if __name__ == "__main__":
    main()
