"""
Tests for partisan_classifier.data.splits.

Once publisher_disjoint_split is implemented, the key property to verify is
that no publisher appears in more than one of the returned splits.
"""

import pandas as pd
import pytest

from partisan_classifier.data.splits import publisher_disjoint_split


def make_df(n_publishers_per_label: int = 10, articles_per_publisher: int = 5) -> pd.DataFrame:
    """
    Build a synthetic dataset with a known number of publishers per label with a fixed number of articles.
    """
    rows = []
    for label in ["left", "center", "right"]:
        for i in range(n_publishers_per_label):
            publisher = f"{label}_publisher_{i}"
            for j in range(articles_per_publisher):
                rows.append({
                    "text": f"article {j} from {publisher}",
                    "publisher": publisher,
                    "label": label,
                })
    return pd.DataFrame(rows)


def test_splits_are_publisher_disjoint():
    df = make_df()
    train_df, val_df, test_df = publisher_disjoint_split(df, seed=42)

    train_pubs = set(train_df["publisher"])
    val_pubs = set(val_df["publisher"])
    test_pubs = set(test_df["publisher"])

    assert not (train_pubs & val_pubs), "Train/val publisher overlap"
    assert not (train_pubs & test_pubs), "Train/test publisher overlap"
    assert not (val_pubs & test_pubs), "Val/test publisher overlap"


def test_splits_cover_every_publisher_exactly_once():
    df = make_df()
    train_df, val_df, test_df = publisher_disjoint_split(df, seed=42)

    all_pubs = set(df["publisher"])
    split_pubs = set(train_df["publisher"]) | set(val_df["publisher"]) | set(test_df["publisher"])

    assert split_pubs == all_pubs, "Every publisher should be present in exactly one split"


def test_splits_preserve_all_rows():
    df = make_df()
    train_df, val_df, test_df = publisher_disjoint_split(df, seed=42)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_splits_are_reproducible_given_same_seed():
    df = make_df()
    train_a, val_a, test_a = publisher_disjoint_split(df, seed=8)
    train_b, val_b, test_b = publisher_disjoint_split(df, seed=8)

    assert set(train_a["publisher"]) == set(train_b["publisher"])
    assert set(val_a["publisher"]) == set(val_b["publisher"])
    assert set(test_a["publisher"]) == set(test_b["publisher"])


def test_different_seeds_can_produce_different_splits():
    df = make_df()
    _, val_a, _ = publisher_disjoint_split(df, seed=1)
    _, val_b, _ = publisher_disjoint_split(df, seed=2)

    # Technically the splits could end up with the same publishers even with different seeds, but this is exceedingly unlikely
    assert set(val_a["publisher"]) != set(val_b["publisher"])


def test_approximate_split_proportions():
    df = make_df(n_publishers_per_label=20, articles_per_publisher=5)
    train_df, val_df, test_df = publisher_disjoint_split(df, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42)

    total = len(df)
    # Allow some tolerance
    assert abs(len(train_df) / total - 0.7) < 0.1
    assert abs(len(val_df) / total - 0.15) < 0.1
    assert abs(len(test_df) / total - 0.15) < 0.1


def test_fractions_must_sum_to_one():
    df = make_df()
    with pytest.raises(AssertionError):
        publisher_disjoint_split(df, train_frac=0.5, val_frac=0.5, test_frac=0.5)


def test_raises_on_empty_dataframe():
    empty_df = pd.DataFrame(columns=["text", "publisher", "label"])
    with pytest.raises(AssertionError):
        publisher_disjoint_split(empty_df)


def test_single_publisher_per_label_entirely_in_one_split():
    """With only one publisher per label, the publisher cannot be divided across splits."""
    df = make_df(n_publishers_per_label=1, articles_per_publisher=20)
    train_df, val_df, test_df = publisher_disjoint_split(df, seed=42)

    for label in ["left", "center", "right"]:
        appearances = sum(
            label in split_df["label"].values
            for split_df in (train_df, val_df, test_df)
        )
        assert appearances == 1, f"Label {label} should appear in exactly one split"