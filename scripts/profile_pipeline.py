"""Where does the time go? Times each pipeline stage on a few real articles and reports the GPU setup.

  python scripts/profile_pipeline.py ../data/articles/articles.jsonl --n 8 \
      --backend fastcoref --voteview ../data/voteview/HSall_members.csv --online

The first article is a warm-up (models load) and is not counted. Use a throwaway --lean-cache path to see the
cold-cache cost, and your real one to see the warm cost. --selftest checks this script without models or network.
"""
from __future__ import annotations

import argparse
import functools
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from partisan_classifier.extraction import claims as C
from partisan_classifier.extraction import coref as R
from partisan_classifier.extraction import lean_resolver as L
from partisan_classifier.extraction import subjects as S

SECONDS: dict[str, float] = defaultdict(float)
CALLS: dict[str, int] = defaultdict(int)


def timed(name: str, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        except Exception:
            CALLS[name + " (errors)"] += 1
            raise
        finally:
            SECONDS[name] += time.perf_counter() - t0
            CALLS[name] += 1
    return wrapper


def environment() -> dict:
    info = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "spacy", "fastcoref", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "not installed"
    try:
        import torch
        info["cuda available"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        info["torch build"] = "CPU-only wheel" if "+cpu" in str(info["torch"]) else str(torch.version.cuda or "no CUDA")
    except Exception:
        pass
    info["pipeline device"] = C._device()
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("articles", help="JSONL with id, text, year")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--backend", default="fastcoref", choices=["fastcoref", "rules"])
    ap.add_argument("--voteview")
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--lean-cache", default="data/lean_cache_profile.jsonl")
    ap.add_argument("--no-stance", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="use instant stand-ins for the models and the network")
    args = ap.parse_args()

    print("environment:")
    for k, v in environment().items():
        print(f"  {k:16} {v}")

    stance_fn = None if args.no_stance else S.absa_stance
    wikidata = L.Wikidata() if args.online else None
    if args.selftest:
        C._classify = lambda texts: ["fact"] * len(texts)
        stance_fn = None if args.no_stance else (lambda pairs: [0.0] * len(pairs))
        if args.online:
            def fake(url):
                time.sleep(0.05)
                return {"search": []}
            wikidata = L.Wikidata(fetch=fake)
    C._classify = timed("subjectivity model", C._classify)
    C._candidates = timed("claim extraction (spaCy + rules)", C._candidates)
    S.find_mentions = timed("mention finding", S.find_mentions)
    R.resolve_claims = timed("coref post-processing", R.resolve_claims)
    R.BACKENDS[args.backend] = timed("coreference model", R.BACKENDS[args.backend])
    if stance_fn is not None:
        stance_fn = timed("stance model", stance_fn)
    if wikidata is not None:
        wikidata._fetch = timed("wikidata HTTP", wikidata._fetch)
    L.LeanResolver.resolve = timed("lean lookups (includes the HTTP above)", L.LeanResolver.resolve)
    resolver = L.LeanResolver(L.Voteview(args.voteview) if args.voteview else None, wikidata, args.lean_cache) \
        if (args.voteview or args.online) else None

    records = []
    with open(args.articles, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) > args.n:
                break
    wall = 0.0
    for i, rec in enumerate(records):
        t0 = time.perf_counter()
        S.analyze_articles([rec["text"][:20000]], args.backend, stance_fn, resolver=resolver,
                           as_of_years=[int(rec["year"]) if rec.get("year") else None])
        dt = time.perf_counter() - t0
        if i == 0:
            SECONDS.clear(); CALLS.clear()       # warm-up article: models were loading
            print(f"\nwarm-up article took {dt:.1f} s (not counted)")
            continue
        wall += dt
    n = len(records) - 1
    print(f"\n{n} articles, {wall:.1f} s total, {wall / max(1, n):.2f} s per article\n")
    print(f"{'stage':44}{'s per article':>14}{'share':>8}{'calls per article':>19}")
    stages = {k: v for k, v in SECONDS.items() if "includes" not in k}
    for name, sec in sorted(SECONDS.items(), key=lambda kv: -kv[1]):
        share = f"{sec / wall:.0%}" if wall else "-"
        print(f"{name:44}{sec / max(1, n):14.2f}{share:>8}{CALLS[name] / max(1, n):19.1f}")
    http = SECONDS.get("wikidata HTTP", 0.0)
    calls = CALLS.get("wikidata HTTP", 0)
    if calls:
        print(f"\nWikidata: {calls / n:.1f} requests per article, {http / calls:.2f} s per request "
              f"({CALLS.get('wikidata HTTP (errors)', 0)} failed)")
    print("\nverdict:")
    if wall and http / wall > 0.4:
        print(f"  network lookups take {http / wall:.0%} of the time. Fewer lookups, a warm --lean-cache, or running "
              "without --online is the fix, not the hardware.")
    model = sum(SECONDS.get(k, 0.0) for k in ("subjectivity model", "coreference model", "stance model"))
    if wall and model / wall > 0.4 and C._device() == -1:
        print(f"  the three models take {model / wall:.0%} of the time and run on the CPU. Install a CUDA build of PyTorch.")
    if wall and model / wall > 0.4 and C._device() == 0:
        print(f"  the models take {model / wall:.0%} of the time even with a GPU: check batch sizes and article length.")


if __name__ == "__main__":
    main()