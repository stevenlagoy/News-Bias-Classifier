"""
Turn an ArticleAnalysis into a fixed-length numeric vector, and run that over a labelled corpus.

    python -m partisan_classifier.extraction.features ../data/articles/articles.jsonl ../data/articles/features.jsonl --backend fastcoref --voteview ../data/voteview/HSall_members.csv --online --save-analysis
"""

from __future__ import annotations
 
import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable
 
from .lean_resolver import LeanResolver, build_resolver, lean_to_score
from .subjects import ArticleAnalysis, Subject, absa_stance, analysis_from_dict, analyze_articles
 
SIDE_BAND = 0.25 # |lean| below this counts as neither left nor right
SPEAKER_TYPES = ["expert", "official", "data", "media", "anonymous", "layperson", "unknown"]
 
FEATURE_NAMES = (
    ["n_claims", "log_n_claims", "attributed_share", "first_person_share",
     "opinion_share_voice", "opinion_share_attr", "personal_opinion_share", "expert_opinion_share",
     "n_primary", "n_primary_lean", "lean_coverage", "lean_mean_primary",
     "voice_slant", "voice_slant_n", "quoted_slant", "quoted_slant_n",
     "voice_stance_left", "voice_stance_right", "quoted_stance_left", "quoted_stance_right",
     "quoted_lean_known_share", "quoted_lean_mean", "quoted_lean_mean_distinct",
     "quoted_left_share", "quoted_right_share"]
    + [f"speaker_{t}_share" for t in SPEAKER_TYPES]
    + [f"src_{sl}_on_{tl}" for sl in ("left", "right") for tl in ("left", "right")]
    + [f"n_src_{sl}_on_{tl}" for sl in ("left", "right") for tl in ("left", "right")]
)
 
 
def _wmean(pairs: list[tuple[float, float]]) -> float:
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total else 0.0
 
 
def _share(n: int, d: int) -> float:
    return n / d if d else 0.0
 
 
def article_features(a: ArticleAnalysis) -> dict[str, float]:
    f = {name: 0.0 for name in FEATURE_NAMES}
    claims = a.claims
    n = len(claims)
    f["n_claims"], f["log_n_claims"] = float(n), math.log1p(n)
 
    voice = [c for c in claims if c.claim.source is None]
    attr = [c for c in claims if c.claim.source is not None]
    f["attributed_share"] = _share(len(attr), n)
    f["first_person_share"] = _share(sum(c.first_person for c in claims), n)
    f["opinion_share_voice"] = _share(sum(c.claim.kind == "opinion" for c in voice), len(voice))
    f["opinion_share_attr"] = _share(sum(c.claim.kind == "opinion" for c in attr), len(attr))
    f["personal_opinion_share"] = _share(sum(c.opinion_type == "personal opinion" for c in attr), len(attr))
    f["expert_opinion_share"] = _share(sum(c.opinion_type.startswith("expert") for c in attr), len(attr))
    for t in SPEAKER_TYPES:
        f[f"speaker_{t}_share"] = _share(sum(c.speaker_type == t for c in attr), len(attr))
 
    # who is quoted
    scored = [(c, lean_to_score(c.source_lean)) for c in attr]
    scored = [(c, s) for c, s in scored if s is not None]
    f["quoted_lean_known_share"] = _share(len(scored), len(attr))
    if scored:
        f["quoted_lean_mean"] = sum(s for _, s in scored) / len(scored)
        by_source: dict[str, float] = {}
        for c, s in scored:
            by_source[c.source_key or c.claim.source or ""] = s
        f["quoted_lean_mean_distinct"] = sum(by_source.values()) / len(by_source)
        f["quoted_left_share"] = _share(sum(s < -SIDE_BAND for _, s in scored), len(scored))
        f["quoted_right_share"] = _share(sum(s > SIDE_BAND for _, s in scored), len(scored))
 
    # how primary subjects are presented
    subs: list[Subject] = a.primary() or a.subjects[:3]
    known = [s for s in subs if s.lean_score is not None]
    f["n_primary"], f["n_primary_lean"] = float(len(subs)), float(len(known))
    f["lean_coverage"] = _share(sum(s.score for s in known), sum(s.score for s in subs)) if subs else 0.0 # type: ignore
 
    def weight(s: Subject) -> float:
        return s.score * max(s.lean_confidence, 0.1)
 
    f["lean_mean_primary"] = _wmean([(s.lean_score, weight(s)) for s in known]) # type: ignore[arg-type]
    for channel in ("voice", "quoted"):
        slant: list[tuple[float, float]] = []
        side: dict[str, list[tuple[float, float]]] = {"left": [], "right": []}
        for s in known:
            ch = s.stance_voice if channel == "voice" else s.stance_quoted
            if ch is None:
                continue
            slant.append((s.lean_score * ch.mean, weight(s))) # type: ignore[operator]
            if s.lean_score < -SIDE_BAND: # type: ignore[operator]
                side["left"].append((ch.mean, weight(s)))
            elif s.lean_score > SIDE_BAND: # type: ignore[operator]
                side["right"].append((ch.mean, weight(s)))
        f[f"{channel}_slant"] = _wmean(slant)
        f[f"{channel}_slant_n"] = float(len(slant))
        f[f"{channel}_stance_left"] = _wmean(side["left"])
        f[f"{channel}_stance_right"] = _wmean(side["right"])
 
    # which side says what about which side
    acc: dict[str, list[tuple[float, float]]] = {k: [] for k in FEATURE_NAMES if k.startswith("src_")}
    for s in known:
        if abs(s.lean_score) < SIDE_BAND: # type: ignore[arg-type]
            continue
        target = "left" if s.lean_score < 0 else "right" # type: ignore[operator]
        for source_lean, ch in s.stance_by_source_lean.items():
            sl = lean_to_score(source_lean)
            if sl is None or abs(sl) < SIDE_BAND:
                continue
            acc[f"src_{'left' if sl < 0 else 'right'}_on_{target}"].append((ch.mean, float(ch.n)))
    for k, pairs in acc.items():
        f[k] = _wmean(pairs)
        f["n_" + k] = sum(w for _, w in pairs)
    return f

def _done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(str(json.loads(line)["id"]))
            except (ValueError, KeyError):
                continue
    return done
 
 
def run_batch(
    records: Iterable[dict], out_path: str | Path, *, backend: str = "fastcoref",
    stance_fn=absa_stance, resolver: LeanResolver | None = None, chunk: int = 16,
    max_chars: int = 20000, id_key: str = "id", text_key: str = "text", year_key: str = "year",
    keep: tuple[str, ...] = ("label", "publisher", "split"), save_analysis: bool = False,
) -> tuple[int, int]:
    """Analyze each record and append {id, kept fields, features, primary subjects} to out_path.
 
    Rerunning skips ids already in the file, so an interrupted run resumes. Failures are logged to
    <out_path>.errors and the run continues. Returns (written, failed).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path = out_path.with_suffix(out_path.suffix + ".errors")
    done = _done_ids(out_path)
    written = failed = 0
 
    def process(batch: list[dict]) -> None:
        nonlocal written, failed
        texts = [str(r[text_key])[:max_chars] for r in batch]
        years = [int(r[year_key]) if r.get(year_key) else None for r in batch]
        try:
            analyses = analyze_articles(texts, backend, stance_fn, resolver=resolver, as_of_years=years)
        except Exception: # one bad article should not sink the chunk: retry one at a time
            analyses = []
            for r, t, y in zip(batch, texts, years):
                try:
                    analyses.append(analyze_articles([t], backend, stance_fn, resolver=resolver, as_of_years=[y])[0])
                except Exception as e:
                    analyses.append(None)
                    failed += 1
                    with open(err_path, "a", encoding="utf-8") as ef:
                        ef.write(json.dumps({"id": r[id_key], "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}) + "\n")
        with open(out_path, "a", encoding="utf-8") as out:
            for r, a in zip(batch, analyses):
                if a is None:
                    continue
                row = {"id": str(r[id_key]), **{k: r.get(k) for k in keep},
                       "features": article_features(a),
                       "primary": [{"key": s.key, "lean": s.political_lean, "lean_score": s.lean_score,
                                    "lean_method": s.lean_method, "score": s.score} for s in a.primary()]}
                if save_analysis:
                    row["analysis"] = a.to_dict()
                out.write(json.dumps(row) + "\n")
                written += 1
            out.flush()
 
    buf: list[dict] = []
    for rec in records:
        if str(rec.get(id_key)) in done:
            continue
        if not str(rec.get(text_key) or "").strip():
            failed += 1
            with open(err_path, "a", encoding="utf-8") as ef:
                ef.write(json.dumps({"id": rec.get(id_key), "error": "missing text"}) + "\n")
            continue
        buf.append(rec)
        if len(buf) >= chunk:
            process(buf)
            print(f"\r{written} written, {failed} failed", end="", file=sys.stderr)
            buf = []
            if written == 0 and failed >= 2 * chunk: # There is something very wrong and it's not worth wasting resources
                sys.exit(f"\nthe first {failed} articles all failed; see {err_path}")
    if buf:
        process(buf)
    print(f"\r{written} written, {failed} failed", file=sys.stderr)
    return written, failed
 

def rebuild_features(in_path: str | Path, out_path: str | Path) -> int:
    """Recompute the feature vectors from cached analyses (rows written with save_analysis=True)."""
    n = 0
    with open(in_path, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            row = json.loads(line)
            if "analysis" not in row:
                continue
            row["features"] = article_features(analysis_from_dict(row["analysis"]))
            out.write(json.dumps(row) + "\n")
            n += 1
    return n


def load_features(path: str | Path, names: list[str] = FEATURE_NAMES):
    """Read a features JSONL into (ids, X, meta) with columns in a fixed order."""
    import numpy as np
    ids, rows, meta = [], [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        ids.append(d["id"])
        rows.append([float(d["features"].get(k, 0.0)) for k in names])
        meta.append({k: v for k, v in d.items() if k not in {"features", "primary"}})
    return ids, np.array(rows, dtype=float), meta
 
 
def main() -> None:
    ap = argparse.ArgumentParser(description="Extract article features for a labeled corpus.")
    ap.add_argument("input", help="JSONL with id, text, and optional label / publisher / year")
    ap.add_argument("output", help="features JSONL (appended to, so reruns resume)")
    ap.add_argument("--backend", default="fastcoref", choices=["fastcoref", "rules"])
    ap.add_argument("--voteview")
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--lean-cache", default="data/lean_cache.jsonl")
    ap.add_argument("--no-stance", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="force model execution onto CPU or CUDA")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--save-analysis", action="store_true", help="keep each article's full analysis in the output")
    ap.add_argument("--rebuild", action="store_true", help="input is a features file saved with --save-analysis; recompute its features (no models run)")
    args = ap.parse_args()
    if args.rebuild:
        print(f"rebuilt {rebuild_features(args.input, args.output)} rows", file=sys.stderr)
        return
 
    def records():
        with open(args.input, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if args.limit and i >= args.limit:
                    return
                yield json.loads(line)
 
    os.environ["PARTISAN_DEVICE"] = args.device
    resolver = build_resolver(args.voteview, args.online, args.lean_cache) if (args.voteview or args.online) else None
    t0 = time.time()
    run_batch(records(), args.output, backend=args.backend, stance_fn=None if args.no_stance else absa_stance, resolver=resolver, save_analysis=args.save_analysis)
    print(f"done in {time.time() - t0:.0f}s", file=sys.stderr)
 
 
if __name__ == "__main__":
    main()
