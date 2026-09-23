"""
Publisher-disjoint train/val/test splitting.

The core requirement for this project: no publisher/outlet may appear in more
than one split, so the model can't learn outlet-specific style as a shortcut
for partisan label.
"""

from __future__ import annotations

import pandas as pd


def publisher_disjoint_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train/val/test with disjoint sets of publishers.

    Splits at the publisher level (not the article level) and assigns whole
    publishers to a split, ideally keeping label balance across splits
    reasonably close to the overall distribution.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, (
        "Split fractions must sum to 1.0"
    )

    # One dominant label per publisher (mode of articles' labels)
    pub_label = df.groupby("publisher")["label"].agg(lambda s: s.mode()[0])

    train_pubs, val_pubs, test_pubs = [], [], []
    for label, group in pub_label.groupby(pub_label):
        pubs = group.index.to_series().sample(frac=1.0, random_state=seed).tolist()
        n = len(pubs)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)
        train_pubs += pubs[:n_train]
        val_pubs += pubs[n_train:n_train + n_val]
        test_pubs += pubs[n_train + n_val:]

    train_df = df[df["publisher"].isin(train_pubs)]
    val_df = df[df["publisher"].isin(val_pubs)]
    test_df = df[df["publisher"].isin(test_pubs)]

    assert not (set(train_pubs) & set(val_pubs)), "Train/val publisher overlap"
    assert not (set(train_pubs) & set(test_pubs)), "Train/test publisher overlap"
    assert not (set(val_pubs) & set(test_pubs)), "Val/test publisher overlap"
    return train_df, val_df, test_df
