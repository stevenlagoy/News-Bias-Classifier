"""Build the JSONL that features.py reads from the Article-Bias-Prediction JSON files.

Each output line: {"id", "text", "label", "publisher", "year", "title", "split"}.

Sampling is balanced by label and spread across publishers (round-robin), so a small run still
covers many outlets. Split files are optional; pass one or more NAME=PATH pairs, where PATH is a TSV/CSV
with an ID column (the dataset's own split files work) or a plain list of ids:

  python scripts/make_articles_jsonl.py --src data/Article-Bias-Prediction/data/jsons \
      --split train=data/splits/media/train.tsv --split test=data/splits/media/test.tsv \
      --per-label 100 --out data/articles/articles.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LABELS = {0: "left", 1: "center", 2: "right"}


def year_of(date) -> int | None:
    m = re.search(r"(?:19|20)\d{2}", str(date or ""))
    return int(m.group()) if m else None


def read_ids(path: str) -> set[str]:
    """Ids from a split file: the 'ID' column if there is a header, else the first column."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    first = Path(path).read_text(encoding="utf-8").splitlines()[0]
    delim = "\t" if "\t" in first else ","
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter=delim, quoting=csv.QUOTE_NONE))
    header = [c.strip().lower() for c in rows[0]]
    if "id" in header:
        col = header.index("id")
        return {r[col].strip() for r in rows[1:] if len(r) > col}
    return {r[0].strip() for r in rows if r}


def balanced_sample(items: list[dict], per_label: int, rng: random.Random) -> list[dict]:
    """Up to per_label articles for each label, taking one per publisher in turn."""
    out: list[dict] = []
    for label in LABELS.values():
        by_pub: dict[str, list[dict]] = defaultdict(list)
        for it in items:
            if it["label"] == label:
                by_pub[it["publisher"]].append(it)
        for v in by_pub.values():
            rng.shuffle(v)
        pubs = sorted(by_pub)
        rng.shuffle(pubs)
        picked: list[dict] = []
        while len(picked) < per_label and any(by_pub[p] for p in pubs):
            for p in pubs:
                if by_pub[p] and len(picked) < per_label:
                    picked.append(by_pub[p].pop())
        out.extend(picked)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory of Article-Bias-Prediction .json files")
    ap.add_argument("--out", default="data/articles/articles.jsonl")
    ap.add_argument("--per-label", type=int, default=100, help="articles per label (per split, if splits are given)")
    ap.add_argument("--split", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=500)
    args = ap.parse_args()

    if not Path(args.src).is_dir():
        sys.exit(f"--src is not a folder: {args.src!r}. Use the real path to the folder of .json files.")
    split_ids = {}
    for spec in args.split:
        name, _, path = spec.partition("=")
        if not Path(path).is_file():
            sys.exit(f"--split {name}: no such file: {path!r}. Use a real path, or leave --split out.")
        split_ids[name] = read_ids(path)
        print(f"split {name}: {len(split_ids[name])} ids", file=sys.stderr)

    articles: list[dict] = []
    for p in sorted(Path(args.src).glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        # content_original keeps the casing and punctuation that NER and quote detection depend on
        text = d.get("content_original") or d.get("content") or ""
        label = LABELS.get(int(d["bias"])) if str(d.get("bias", "")).strip() in {"0", "1", "2"} else None
        if label is None or len(text) < args.min_chars:
            continue
        articles.append({"id": str(d.get("ID") or p.stem), "text": text, "label": label,
                         "publisher": d.get("source") or "unknown", "year": year_of(d.get("date")),
                         "title": d.get("title")})
    print(f"{len(articles)} usable articles", file=sys.stderr)
    have = {a["id"] for a in articles}
    for name, ids in split_ids.items():
        matched = len(ids & have)
        print(f"split {name}: {matched} of {len(ids)} ids match an article", file=sys.stderr)
        if matched == 0:
            sys.exit(f"split {name}: no ids matched. Check that the file's ID column holds the same ids as the .json files.")

    rng = random.Random(args.seed)
    chosen: list[dict] = []
    if split_ids:
        for name, ids in split_ids.items():
            pool = [a for a in articles if a["id"] in ids]
            for a in balanced_sample(pool, args.per_label, rng):
                chosen.append({**a, "split": name})
    else:
        chosen = [{**a, "split": None} for a in balanced_sample(articles, args.per_label, rng)]

    rng.shuffle(chosen)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for a in chosen:
            f.write(json.dumps(a) + "\n")
    print(f"wrote {len(chosen)} articles to {args.out}", file=sys.stderr)
    for key in sorted({a["split"] or "-" for a in chosen}):
        part = [a for a in chosen if (a["split"] or "-") == key]
        print(f"  {key}: {dict(Counter(a['label'] for a in part))}, {len({a['publisher'] for a in part})} publishers",
              file=sys.stderr)


if __name__ == "__main__":
    main()