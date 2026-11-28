"""Test whether structured article features add value beyond a TF-IDF text model.

Example:
    python scripts/stack_features.py --src data/article-bias-prediction/data/jsons ^
        --train-ids data/article-bias-prediction/data/splits/media/train.tsv ^
        --features data/articles/features_v5.jsonl

The feature file should contain train/valid/test rows. Text-model predictions
for feature-train rows are publisher-grouped out-of-fold predictions. The
TF-IDF vectorizer is fitted separately inside each fold, so the held-out
publisher's text is not used to learn vocabulary or IDF weights.

Feature-valid/test article IDs are explicitly removed from the text training
corpus, preventing accidental article-level leakage if those IDs also appear
in the supplied --train-ids file. A final text model is fitted on the remaining
text corpus for validation/test predictions.

The meta-model is trained on feature-train rows only, with publisher-grouped
cross-validation selecting regularization. Results are reported separately
for all held-out articles and publishers absent from the text training corpus.
Paired publisher-cluster bootstrap intervals are reported for both accuracy
and macro-F1 differences.
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from partisan_classifier.extraction.features import FEATURE_NAMES, load_features

CLASSES = ["center", "left", "right"]
LABELS = {0: "left", 1: "center", 2: "right"}
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


def norm_pub(name: str) -> str:
    return re.sub(r"^the", "", re.sub(r"[^a-z0-9]", "", str(name).lower()))


def read_ids(path: str) -> set[str]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    first = Path(path).read_text(encoding="utf-8").splitlines()[0]
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(
            f,
            delimiter="\t" if "\t" in first else ",",
            quoting=csv.QUOTE_NONE,
        ))
    if not rows:
        return set()
    header = [c.strip().lower() for c in rows[0]]
    if "id" in header:
        col = header.index("id")
        return {r[col].strip() for r in rows[1:] if len(r) > col}
    return {r[0].strip() for r in rows if r and r[0].strip()}


def load_articles(src: str, ids: set[str], field: str) -> dict[str, dict]:
    out = {}
    for path in Path(src).glob("*.json"):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        article_id = str(d.get("ID") or path.stem)
        if article_id not in ids:
            continue
        if str(d.get("bias", "")).strip() not in {"0", "1", "2"}:
            continue
        text = d.get(field) or d.get("content") or d.get("content_original") or ""
        if text:
            out[article_id] = {
                "text": text,
                "label": LABELS[int(d["bias"])],
                "publisher": d.get("source") or "unknown",
            }
    return out


def make_vectorizer(max_features: int):
    return TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=3,
        stop_words="english",
    )


def text_model(max_features: int):
    return make_pipeline(
        make_vectorizer(max_features),
        LogisticRegression(max_iter=1500, C=10, class_weight="balanced"),
    )


def class_order(model, probabilities):
    order = [list(model.classes_).index(c) for c in CLASSES]
    return probabilities[:, order]


def grouped_splits(groups: np.ndarray):
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("At least two publishers are needed for grouped cross-validation.")
    return list(GroupKFold(n_splits=min(5, n_groups)).split(
        np.zeros(len(groups)), groups, groups
    ))


def meta_model(Xtr, ytr, groups):
    cv = grouped_splits(groups)
    search = GridSearchCV(
        make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        ),
        {"logisticregression__C": C_GRID},
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    return search.fit(Xtr, ytr)


def metric_pair(y, pred):
    return (
        f1_score(y, pred, average="macro", zero_division=0),
        accuracy_score(y, pred),
    )


def paired_cluster_ci(y, pred_a, pred_b, groups, rng, n_boot=2000):
    """CI for metric(A)-metric(B), resampling publishers and retaining pairs."""
    members = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    keys = np.array(list(members), dtype=object)
    f1_diffs, acc_diffs = [], []
    for _ in range(n_boot):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([members[g] for g in sampled])
        f1_diffs.append(
            f1_score(y[idx], pred_a[idx], average="macro", zero_division=0)
            - f1_score(y[idx], pred_b[idx], average="macro", zero_division=0)
        )
        acc_diffs.append(
            accuracy_score(y[idx], pred_a[idx])
            - accuracy_score(y[idx], pred_b[idx])
        )
    return (
        np.percentile(f1_diffs, [2.5, 97.5]),
        np.percentile(acc_diffs, [2.5, 97.5]),
    )


def report_group(name, mask, y, groups, predictions, rng, n_boot):
    n_publishers = len(np.unique(groups[mask]))
    print(f"\n{name}: {int(mask.sum())} articles from {n_publishers} publishers")
    print(f"  {'model':27s}{'macro-F1':>10s}{'accuracy':>10s}")
    for model_name, pred in predictions.items():
        f1, acc = metric_pair(y[mask], pred[mask])
        print(f"  {model_name:27s}{f1:10.3f}{acc:10.3f}")

    if n_publishers < 2:
        print("  Too few publishers for a cluster-bootstrap interval.")
        return

    for a_name, b_name in (
        ("text + features", "text only (TF-IDF)"),
        ("features only", "text only (TF-IDF)"),
    ):
        f1_ci, acc_ci = paired_cluster_ci(
            y[mask], predictions[a_name][mask], predictions[b_name][mask],
            groups[mask], rng, n_boot,
        )
        f1_diff, acc_diff = (
            metric_pair(y[mask], predictions[a_name][mask])[0]
            - metric_pair(y[mask], predictions[b_name][mask])[0],
            metric_pair(y[mask], predictions[a_name][mask])[1]
            - metric_pair(y[mask], predictions[b_name][mask])[1],
        )
        f1_word = "better" if f1_ci[0] > 0 else "worse" if f1_ci[1] < 0 else "not distinguishable"
        acc_word = "better" if acc_ci[0] > 0 else "worse" if acc_ci[1] < 0 else "not distinguishable"
        print(
            f"  {a_name} minus {b_name}:\n"
            f"    macro-F1 {f1_diff:+.3f} (95% CI {f1_ci[0]:+.3f} to {f1_ci[1]:+.3f}): {f1_word}\n"
            f"    accuracy {acc_diff:+.3f} (95% CI {acc_ci[0]:+.3f} to {acc_ci[1]:+.3f}): {acc_word}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder containing Article-Bias-Prediction JSON files")
    ap.add_argument("--train-ids", required=True, help="official text-model training IDs")
    ap.add_argument("--features", required=True, help="features JSONL with train/valid/test split")
    ap.add_argument("--text-field", default="content", help="article JSON text field")
    ap.add_argument("--max-features", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    ids, X, meta = load_features(args.features)
    y = np.array([m["label"] for m in meta])
    split = np.array([str(m.get("split") or "").strip().lower() for m in meta])
    pub_raw = np.array([str(m.get("publisher") or "unknown") for m in meta])
    pub_norm = np.array([norm_pub(p) for p in pub_raw])

    train_rows = np.flatnonzero(split == "train")
    valid_rows = np.flatnonzero(split == "valid")
    test_rows = np.flatnonzero(split == "test")
    if not len(train_rows) or not len(test_rows):
        raise ValueError(
            "Expected feature rows with split='train' and split='test'. "
            "A validation split is optional."
        )
    if len(set(ids)) != len(ids):
        raise ValueError("Feature file contains duplicate article IDs; resolve duplicates before stacking.")

    feature_train_ids = {ids[i] for i in train_rows}
    heldout_ids = {ids[i] for i in np.concatenate([valid_rows, test_rows])}

    official_ids = read_ids(args.train_ids)
    # Never let a validation/test article enter the text model's fitting corpus,
    # even if the supplied official train-ID list happens to contain it.
    corpus_ids = official_ids - heldout_ids
    articles = load_articles(args.src, corpus_ids, args.text_field)
    if not articles:
        raise ValueError("No usable text articles loaded. Check --src, --train-ids, and --text-field.")

    text_ids = sorted(articles)
    text = [articles[i]["text"] for i in text_ids]
    text_y = np.array([articles[i]["label"] for i in text_ids])
    text_pub = np.array([norm_pub(articles[i]["publisher"]) for i in text_ids])
    text_pos = {article_id: i for i, article_id in enumerate(text_ids)}
    text_publishers = set(text_pub)

    print(f"Feature rows: {len(ids)} ({len(train_rows)} train, {len(valid_rows)} valid, {len(test_rows)} test)")
    print(f"Text corpus: {len(text_ids)} articles from {len(text_publishers)} publishers")
    print(f"Feature-train articles present in text corpus: {len(feature_train_ids & set(text_ids))}/{len(feature_train_ids)}")
    print(f"Held-out feature IDs excluded from text corpus: {len(heldout_ids & official_ids)}")

    # Publisher-held-out OOF probabilities for feature-training rows.
    train_feature_rows = [
        i for i in train_rows if ids[i] in text_pos
    ]
    if not train_feature_rows:
        raise ValueError(
            "None of the feature-train article IDs occur in the text training corpus. "
            "The meta-model cannot be trained with publisher-held-out text predictions."
        )

    oof_prob: dict[str, np.ndarray] = {}
    splits = grouped_splits(text_pub)
    for fold, (fit_idx, held_idx) in enumerate(splits, start=1):
        model = text_model(args.max_features)
        model.fit([text[i] for i in fit_idx], text_y[fit_idx])
        held_ids = {text_ids[i] for i in held_idx}
        needed = [i for i in train_feature_rows if ids[i] in held_ids]
        if needed:
            rows = [text_pos[ids[i]] for i in needed]
            probs = class_order(
                model,
                model.predict_proba([text[r] for r in rows]),
            )
            for feature_row, probability in zip(needed, probs):
                oof_prob[ids[feature_row]] = probability
        print(f"  OOF text fold {fold}/{len(splits)} complete")

    meta_train_rows = np.array([i for i in train_feature_rows if ids[i] in oof_prob])
    if len(meta_train_rows) < 10:
        raise ValueError(f"Only {len(meta_train_rows)} feature-train rows have OOF text predictions.")

    # Fit one final text model on the permitted text corpus. Held-out feature
    # articles were excluded above, so these predictions are article-disjoint.
    final_text = text_model(args.max_features)
    final_text.fit(text, text_y)

    def predict_text(rows):
        valid_rows_with_text = [i for i in rows if ids[i] in load_test_articles]
        if not valid_rows_with_text:
            return np.empty((0, len(CLASSES))), np.array([], dtype=int)
        probabilities = class_order(
            final_text,
            final_text.predict_proba([load_test_articles[ids[i]]["text"] for i in valid_rows_with_text]),
        )
        return probabilities, np.array(valid_rows_with_text, dtype=int)

    # Load validation/test article text independently of the training IDs.
    load_test_articles = load_articles(args.src, heldout_ids, args.text_field)
    missing = [ids[i] for i in np.concatenate([valid_rows, test_rows]) if ids[i] not in load_test_articles]
    if missing:
        print(f"Warning: {len(missing)} validation/test feature rows have no article text and will be omitted.")

    text_prob_by_id: dict[str, np.ndarray] = {}
    for rows in (valid_rows, test_rows):
        probs, used_rows = predict_text(rows)
        for row, probability in zip(used_rows, probs):
            text_prob_by_id[ids[row]] = probability

    # Construct aligned matrices; only rows with usable text can be evaluated.
    train_prob = np.array([oof_prob[ids[i]] for i in meta_train_rows])
    train_y = y[meta_train_rows]
    train_groups = pub_norm[meta_train_rows]
    if len(np.unique(train_groups)) < 2:
        raise ValueError("Feature-training rows contain fewer than two publishers.")

    text_train = np.log(train_prob + 1e-6)
    X_train_meta = np.hstack([text_train, X[meta_train_rows]])

    # Train feature-only and combined meta-models using the same rows and
    # publisher-grouped CV procedure.
    feature_model = meta_model(X[meta_train_rows], train_y, train_groups)
    combined_model = meta_model(X_train_meta, train_y, train_groups)
    print(f"Meta training: {len(meta_train_rows)} articles from {len(np.unique(train_groups))} publishers")
    print(f"Selected C: features={feature_model.best_params_['logisticregression__C']}, "
          f"combined={combined_model.best_params_['logisticregression__C']}")

    # Build prediction arrays indexed by the original feature-file row.
    predictions = {
        "text only (TF-IDF)": np.full(len(ids), "", dtype=object),
        "features only": np.full(len(ids), "", dtype=object),
        "text + features": np.full(len(ids), "", dtype=object),
    }
    # Text-only predictions use the final text model.
    for row_id, probability in text_prob_by_id.items():
        row = ids.index(row_id)
        predictions["text only (TF-IDF)"][row] = CLASSES[int(np.argmax(probability))]
    # Feature/meta predictions require corresponding text probabilities for combined model.
    for row in np.concatenate([valid_rows, test_rows]):
        if ids[row] not in text_prob_by_id:
            continue
        predictions["features only"][row] = feature_model.predict(X[row:row + 1])[0]
        combined_input = np.hstack([
            np.log(text_prob_by_id[ids[row]] + 1e-6).reshape(1, -1),
            X[row:row + 1],
        ])
        predictions["text + features"][row] = combined_model.predict(combined_input)[0]

    seen_in_text_train = np.array([p in text_publishers for p in pub_norm])
    for split_name, rows in (("valid", valid_rows), ("test", test_rows)):
        if not len(rows):
            continue
        usable = np.array([
            i for i in rows
            if predictions["text only (TF-IDF)"][i] != ""
            and predictions["features only"][i] != ""
            and predictions["text + features"][i] != ""
        ], dtype=int)
        if not len(usable):
            print(f"\n{split_name}: no rows had both labels and usable article text.")
            continue
        y_eval = y[usable]
        groups_eval = pub_norm[usable]
        pred_eval = {name: pred[usable] for name, pred in predictions.items()}
        report_group(
            f"{split_name}: all publishers",
            np.ones(len(usable), dtype=bool),
            y_eval, groups_eval, pred_eval, rng, args.bootstrap,
        )
        new_mask = ~seen_in_text_train[usable]
        if np.any(new_mask):
            report_group(
                f"{split_name}: publishers absent from text training",
                new_mask,
                y_eval, groups_eval, pred_eval, rng, args.bootstrap,
            )
        else:
            print(f"\n{split_name}: no publishers absent from the text training corpus.")

    print("\nNotes:")
    print("  * OOF TF-IDF predictions for meta-training rows come from models whose")
    print("    vectorizer and classifier were both fitted without that publisher.")
    print("  * Validation/test articles are excluded from the text fitting corpus.")
    print("  * 'New publisher' means its normalized publisher name is absent from")
    print("    the text corpus; this is a name-based proxy, not entity resolution.")
    print("  * The reported bootstrap intervals resample publishers, not individual articles.")


if __name__ == "__main__":
    main()
