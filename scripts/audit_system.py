"""Audit the system against its goals: measure what can be measured, list what is missing.

  python scripts/audit_system.py --features data/articles/features_v3.jsonl \
      --baseline tfidf=0.525 --baseline politicalBiasBERT=0.534

  python scripts/audit_system.py --features ... --gold-sample 60      # writes outputs/gold_sheet.csv to hand-label
  python scripts/audit_system.py --score-gold outputs/gold_sheet.csv  # accuracy of the extraction, from your labels

The features file must have been written with --save-analysis. Output goes to the console and to
outputs/audit_report.md. The pass bars are the BARS dict below: edit them to match what you decide counts as done.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from partisan_classifier.extraction import explain as E
from partisan_classifier.extraction import features as F
from partisan_classifier.extraction.lean_resolver import lean_to_score
from partisan_classifier.extraction.subjects import analysis_from_dict

BARS = {
    "failure_rate_max": 0.01,            # share of articles that errored out
    "articles_with_lean_min": 0.70,      # share of articles with a primary subject whose lean is known
    "official_lean_min": 0.70,           # share of claims by officials (politicians) whose source lean is known
    "pronoun_source_max": 0.10,          # share of attributed claims whose source is still "he"/"she"/...
    "lowercase_name_share": 0.15,        # a subject mostly matched as a lowercase word ("state", "government") is an alias bug
    "faithfulness_share_min": 0.70,      # top-k deletion must hurt more than random deletion this often
    "gold_source_min": 0.85,             # hand-labeled accuracy of claim sources
    "gold_stance_min": 0.75,             # hand-labeled accuracy of stance scores
}
PRONOUNS = {"he", "she", "they", "it", "i", "we", "you", "him", "her", "them"}
CLASSES = ["center", "left", "right"]


class Report:
    def __init__(self):
        self.items: list[tuple[str, str, str]] = []   # (status, check, detail)
        self.lines: list[str] = []

    def section(self, title: str):
        print(f"\n== {title}")
        self.lines += ["", f"## {title}", ""]

    def say(self, text: str):
        print(text)
        self.lines.append(text)

    def check(self, status: str, name: str, detail: str):
        self.items.append((status, name, detail))
        print(f"[{status:12}] {name}: {detail}")
        self.lines.append(f"- **{status}** {name}: {detail}")


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


class Ctx:
    pass


def load(path: str) -> Ctx:
    c = Ctx()
    c.path = path
    c.rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    c.names = F.FEATURE_NAMES
    c.X = np.array([[r["features"].get(k, 0.0) for k in c.names] for r in c.rows])
    c.y = np.array([r.get("label") for r in c.rows])
    c.pub = np.array([r.get("publisher") or "" for r in c.rows])
    c.split = np.array([r.get("split") or "" for r in c.rows])
    c.has_analysis = all("analysis" in r for r in c.rows)
    c.fold_models, c.fold_of = [], np.zeros(len(c.rows), dtype=int)
    return c


# --------------------------------------------------------------------------------------
def check_health(ctx: Ctx, R: Report):
    R.section("1. Does it run on real articles?")
    err_path = Path(ctx.path + ".errors")
    errors = [json.loads(line) for line in open(err_path, encoding="utf-8")] if err_path.exists() else []
    ok_ids = {r["id"] for r in ctx.rows}
    errors = [e for e in errors if str(e.get("id")) not in ok_ids]     # an id that later succeeded is not a failure
    rate = len(errors) / max(1, len(errors) + len(ctx.rows))
    R.check("PASS" if rate <= BARS["failure_rate_max"] else "FAIL", "failure rate",
            f"{len(errors)} of {len(errors) + len(ctx.rows)} articles failed ({rate:.1%}); bar {BARS['failure_rate_max']:.0%}")
    for msg, n in Counter(e.get("error", "")[:90] for e in errors).most_common(3):
        R.say(f"    {n}x {msg}")
    R.check("PASS" if ctx.has_analysis else "FAIL", "analysis cached",
            "every row has its full analysis" if ctx.has_analysis else "rerun features.py with --save-analysis")
    by_split = Counter(ctx.split.tolist())
    R.say(f"    articles: {len(ctx.rows)}, publishers: {len(set(ctx.pub))}, splits: {dict(by_split)}")


def check_coverage(ctx: Ctx, R: Report):
    R.section("2. Are the pieces finding things? (coverage and hygiene)")
    prim = [r.get("primary", []) for r in ctx.rows]
    with_lean = np.mean([any(p["lean"] for p in ps) for ps in prim])
    R.check("PASS" if with_lean >= BARS["articles_with_lean_min"] else "FAIL", "articles with a known-lean primary subject",
            f"{with_lean:.0%}; bar {BARS['articles_with_lean_min']:.0%}")
    methods = Counter((p["lean_method"] or "none") for ps in prim for p in ps)
    R.say(f"    lean source of primary subjects: {dict(methods)}")
    unknown = Counter(p["key"] for ps in prim for p in ps if not p["lean"])
    R.say(f"    most frequent primary subjects with no lean (candidates for SUBJECTS_ALIGNMENTS): "
          f"{[(k, n) for k, n in unknown.most_common(15)]}")
    if not ctx.has_analysis:
        return
    claims = [c for r in ctx.rows for c in r["analysis"]["claims"]]
    attr = [c for c in claims if c["claim"]["source"]]
    R.say(f"    claims: {len(claims)} ({len(attr)} attributed)")
    by_type: dict[str, list[bool]] = defaultdict(list)
    for c in attr:
        by_type[c["speaker_type"]].append(bool(c["source_lean"]))
    R.say("    source lean known, by speaker type: " + ", ".join(
        f"{t} {np.mean(v):.0%} of {len(v)}" for t, v in sorted(by_type.items(), key=lambda kv: -len(kv[1]))))
    off = np.mean(by_type["official"]) if by_type["official"] else 0.0
    R.check("PASS" if off >= BARS["official_lean_min"] else "FAIL", "claims by officials with a known source lean",
            f"{off:.0%}; bar {BARS['official_lean_min']:.0%}")
    top_unknown = Counter(c["claim"]["source"] for c in attr if c["speaker_type"] == "official" and not c["source_lean"])
    R.say(f"    officials still without a lean: {top_unknown.most_common(12)}")
    pron = np.mean([c["claim"]["source"].strip().lower() in PRONOUNS for c in attr]) if attr else 0.0
    R.check("PASS" if pron <= BARS["pronoun_source_max"] else "FAIL", "sources still a bare pronoun after coref",
            f"{pron:.1%}; bar {BARS['pronoun_source_max']:.0%}")
    unk = np.mean([c["speaker_type"] == "unknown" for c in attr]) if attr else 0.0
    R.say(f"    attributed claims with speaker type 'unknown': {unk:.0%}")
    unscored = Counter(p["lean"] for r in ctx.rows for p in r.get("primary", []) if p["lean"] and p.get("lean_score") is None)
    R.check("PASS" if not unscored else "FAIL", "primary subjects whose lean label has no numeric score",
            "none" if not unscored else f"{dict(unscored)}: these are silently dropped from every lean feature "
            "(check LEAN_SCORE in lean_resolver.py)")
    unscored_src = Counter(c["source_lean"] for c in attr if c["source_lean"] and lean_to_score(c["source_lean"]) is None)
    R.check("PASS" if not unscored_src else "FAIL", "source lean labels with no numeric score",
            "none" if not unscored_src else str(dict(unscored_src)))
    seen = Counter(); names: dict[str, Counter] = defaultdict(Counter)
    for r in ctx.rows:
        for s_ in r["analysis"]["subjects"]:
            seen[s_["key"]] += 1
            for n in s_["names"]:
                names[s_["key"]][n.lower().startswith("the ") and n[4:] or n] += 1
    suspicious = []
    for k, total in names.items():
        low = sum(n for form, n in total.items() if form[:1].islower())
        if sum(total.values()) >= 10 and low / sum(total.values()) > BARS["lowercase_name_share"]:
            suspicious.append((k, f"{low}/{sum(total.values())} lowercase", total.most_common(2)))
    R.check("PASS" if not suspicious else "FAIL", "subjects matched mostly as lowercase words (alias bugs look like this)",
            "none" if not suspicious else str(sorted(suspicious, key=lambda t: -int(t[1].split('/')[0]))[:6]))
    common = [(k, n) for k, n in seen.most_common(8)]
    R.say(f"    most frequent subjects (check they are real): {common}")


def oof_predictions(ctx: Ctx):
    n = len(ctx.y)
    oof = np.zeros((n, len(CLASSES)))
    f1s = []
    ctx.fold_models = []
    for fi, (a, b) in enumerate(GroupKFold(5).split(ctx.X, ctx.y, ctx.pub)):
        m = logreg().fit(ctx.X[a], ctx.y[a])
        order = [list(m.classes_).index(c) for c in CLASSES]
        oof[b] = m.predict_proba(ctx.X[b])[:, order]
        f1s.append(f1_score(ctx.y[b], np.array(CLASSES)[oof[b].argmax(1)], average="macro"))
        ctx.fold_models.append((m, order))
        ctx.fold_of[b] = fi
    return oof, f1s


def check_predictive(ctx: Ctx, R: Report, baselines: dict[str, float]):
    R.section("3. Does the score predict partisanship? (publisher-grouped, out-of-fold)")
    oof, f1s = oof_predictions(ctx)
    mean, se = float(np.mean(f1s)), float(np.std(f1s) / math.sqrt(len(f1s)))
    R.check("PASS" if mean - 2 * se > 1 / 3 else "FAIL", "3-class macro-F1, 5-fold grouped CV",
            f"{mean:.3f} ± {se:.3f} (chance about 0.333)")
    right, left = CLASSES.index("right"), CLASSES.index("left")
    score = oof[:, right] - oof[:, left]
    lr = np.isin(ctx.y, ["left", "right"])
    auc = roc_auc_score(ctx.y[lr] == "right", score[lr])
    rng = np.random.default_rng(0)
    members = {g: np.where(ctx.pub[lr] == g)[0] for g in np.unique(ctx.pub[lr])}
    keys, boots = list(members), []
    ylr, slr = ctx.y[lr] == "right", score[lr]
    for _ in range(300):
        idx = np.concatenate([members[k] for k in rng.choice(keys, len(keys))])
        if len(set(ylr[idx])) == 2:
            boots.append(roc_auc_score(ylr[idx], slr[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    R.check("PASS" if lo > 0.5 else "FAIL", "left vs right AUC of the numeric score",
            f"{auc:.3f} (95% CI {lo:.3f} to {hi:.3f}, resampling {len(keys)} publishers; 0.5 is chance)")
    ordinal = np.array([{"left": -1, "center": 0, "right": 1}.get(v, np.nan) for v in ctx.y])
    ok = ~np.isnan(ordinal)
    rho, p = spearmanr(score[ok], ordinal[ok])
    means = {c: float(score[ctx.y == c].mean()) for c in ("left", "center", "right")}
    R.check("PASS" if rho > 0 and p < 0.01 else "FAIL", "numeric score vs ordinal label (Spearman)",
            f"rho {rho:+.3f}, p {p:.1g}; mean score by label {({k: round(v, 3) for k, v in means.items()})}")
    if (ctx.split == "train").any() and (ctx.split == "test").any():
        m = logreg().fit(ctx.X[ctx.split == "train"], ctx.y[ctx.split == "train"])
        te = ctx.split == "test"
        f1 = f1_score(ctx.y[te], m.predict(ctx.X[te]), average="macro")
        R.say(f"    official split: train on train, test macro-F1 {f1:.3f} ({len(set(ctx.pub[te]))} test publishers)")
        for name, v in baselines.items():
            R.say(f"      versus {name}: {v:.3f} ({f1 - v:+.3f})")
    ctx.oof = oof


def check_explanations(ctx: Ctx, R: Report, n_articles: int, k_max: int, seed: int):
    R.section("4. Can it say what drives a score? (perturbation test)")
    if not ctx.has_analysis:
        R.check("NOT MEASURED", "attribution faithfulness", "needs --save-analysis")
        return
    rng = random.Random(seed)
    order = list(range(len(ctx.rows)))
    rng.shuffle(order)

    def vec(a):
        f = F.article_features(a)
        return np.array([[f.get(k, 0.0) for k in ctx.names]])

    mismatch, d_top, d_rand, wins, used = 0, [], [], 0, 0
    for i in order:
        if used >= n_articles:
            break
        a = analysis_from_dict(ctx.rows[i]["analysis"])
        n = len(a.claims)
        if n < 8:
            continue
        full = E.reaggregate(a, list(range(n)))
        diff = np.abs(vec(full) - ctx.X[i:i + 1]).max()
        mismatch += diff > 0.02
        model, order_idx = ctx.fold_models[ctx.fold_of[i]]
        probs = model.predict_proba(vec(a))[0][order_idx]
        c = int(probs.argmax())

        def predict(x, model=model, order_idx=order_idx, c=c):
            return float(model.predict_proba(vec(x))[0][order_idx][c])

        contrib = E.claim_contributions(a, predict)
        k = max(1, min(k_max, n // 5))
        top = set(np.argsort(contrib)[::-1][:k].tolist())
        p0 = predict(a)
        p_top = predict(E.reaggregate(a, [j for j in range(n) if j not in top]))
        rand = [predict(E.reaggregate(a, [j for j in range(n) if j not in set(rng.sample(range(n), k))])) for _ in range(5)]
        d_top.append(p0 - p_top)
        d_rand.append(p0 - float(np.mean(rand)))
        wins += d_top[-1] > d_rand[-1]
        used += 1
    R.check("PASS" if mismatch <= 0.05 * used else "FAIL", "re-aggregation reproduces the stored features",
            f"{mismatch} of {used} articles differ by more than 0.02 (the cache and the stored features were produced "
            "by different logic, for example a lean label with no numeric score; rerun after fixing)")
    share = wins / max(1, used)
    R.check("PASS" if share >= BARS["faithfulness_share_min"] and np.mean(d_top) > np.mean(d_rand) else "FAIL",
            "removing the claims the explainer picks hurts the prediction more than removing random claims",
            f"top-k drop {np.mean(d_top):.3f} vs random {np.mean(d_rand):.3f}; top-k larger in {share:.0%} of {used} articles")
    a = analysis_from_dict(ctx.rows[order[0]]["analysis"])
    gt = E.ground_truth(a)
    R.check("PASS" if gt["facts"] or gt["attributed_facts"] else "FAIL", "ground-truth export",
            f"{len(gt['facts'])} facts, {len(gt['attributed_facts'])} attributed facts, {len(gt['attributed_statements'])} "
            f"attributed statements kept, {gt['editorial_opinions_removed']} editorial opinions removed")


def has(module: str, attr: str) -> bool:
    try:
        return hasattr(importlib.import_module(module), attr)
    except Exception:
        return False


def check_capabilities(ctx: Ctx, R: Report):
    R.section("5. Which capabilities from the goal exist?")
    table_axes = False
    for modname in ("partisan_classifier.extraction.subjects_politics", "partisan_classifier.extraction.alignments"):
        try:
            tbl = importlib.import_module(modname).SUBJECTS_ALIGNMENTS
            table_axes = any("axes" in v for v in tbl.values())
        except Exception:
            continue
    caps = [
        ("claims with sources, fact/opinion, coref", has("partisan_classifier.extraction.claims", "extract_many")),
        ("primary subjects with stance and lean", has("partisan_classifier.extraction.subjects", "analyze_articles")),
        ("lean lookup for unknown subjects", has("partisan_classifier.extraction.lean_resolver", "LeanResolver")),
        ("numeric score from features (evaluation only)", True),
        ("explanation: which claims drive the score", has("partisan_classifier.extraction.explain", "claim_contributions")),
        ("ground truth with opinion removed", has("partisan_classifier.extraction.explain", "ground_truth")),
        ("saved scorer and a command to score one new article", (ROOT / "scripts" / "score_article.py").exists()),
        ("combined with the text model (stacking)", (ROOT / "scripts" / "stack_features.py").exists()),
        ("more than one axis (economic, social, interventionism...)", table_axes),
        ("reverse direction: rewrite ground truth to a target lean", has("partisan_classifier.extraction.rewrite", "apply_lean")),
        ("reverse direction scored by an independent model", (ROOT / "scripts" / "reverse_test.py").exists()),
    ]
    for name, present in caps:
        R.check("PASS" if present else "TODO", name, "present" if present else "missing")


def make_gold_sheet(ctx: Ctx, n: int, out: Path, seed: int):
    rng = random.Random(seed)
    attrib, stance = [], []
    for r in ctx.rows:
        for c in r["analysis"]["claims"]:
            if c["claim"]["source"]:
                attrib.append((r["id"], c))
            for key, sc in c["targets"]:
                stance.append((r["id"], c, key, sc))
    rng.shuffle(attrib); rng.shuffle(stance)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "article_id", "claim_text", "extracted_source", "source_role", "speaker_type", "opinion_type",
                    "source_lean", "subject", "stance_score", "source_ok", "type_ok", "lean_ok", "stance_ok"])
        for aid, c in attrib[:n]:
            w.writerow(["attribution", aid, c["claim"]["text"], c["claim"]["source"], c["claim"]["source_role"] or "",
                        c["speaker_type"], c["opinion_type"], c["source_lean"] or "", "", "", "", "", "", ""])
        for aid, c, key, sc in stance[:n]:
            w.writerow(["stance", aid, c["claim"]["text"], c["claim"]["source"] or "", "", "", "", "", key, sc, "", "", "", ""])
    print(f"wrote {out}. Fill the blank *_ok columns with 1 (correct) or 0 (wrong), then run --score-gold {out}")
    print("  source_ok: is extracted_source who the article attributes this to? type_ok: are speaker_type and opinion_type right?")
    print("  lean_ok: is source_lean right for that source (leave blank if unknown)?  stance_ok: does stance_score's sign fit the claim toward the subject?")


def score_gold(path: Path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    bars = {"source_ok": BARS["gold_source_min"], "type_ok": 0.80, "lean_ok": 0.80, "stance_ok": BARS["gold_stance_min"]}
    print(f"hand-labeled accuracy from {path}")
    for col, bar in bars.items():
        vals = [r[col].strip() for r in rows if r.get(col, "").strip() in {"0", "1"}]
        k, n = sum(v == "1" for v in vals), len(vals)
        if n == 0:
            print(f"  {col:10} no labels yet")
            continue
        lo, hi = wilson(k, n)
        verdict = "PASS" if lo >= bar else "BELOW BAR" if hi < bar else "UNCERTAIN, label more rows"
        print(f"  {col:10} {k}/{n} = {k / n:.0%} (95% interval {lo:.0%} to {hi:.0%}); bar {bar:.0%}: {verdict}")


# --------------------------------------------------------------------------------------
PRIORITY = {"FAIL": 0, "TODO": 1, "NOT MEASURED": 2, "WARN": 3}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features")
    ap.add_argument("--baseline", action="append", default=[], metavar="NAME=MACRO_F1")
    ap.add_argument("--explain-n", type=int, default=60, help="articles in the perturbation test")
    ap.add_argument("--explain-k", type=int, default=3, help="most claims removed per article")
    ap.add_argument("--gold-sample", type=int, metavar="N", help="write a sheet of N attributions and N stances to hand-label")
    ap.add_argument("--score-gold", metavar="CSV")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "audit_report.md"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.score_gold:
        score_gold(Path(args.score_gold))
        return
    if not args.features:
        ap.error("--features is required")
    ctx = load(args.features)
    if args.gold_sample:
        make_gold_sheet(ctx, args.gold_sample, ROOT / "outputs" / "gold_sheet.csv", args.seed)
        return
    baselines = {k: float(v) for k, _, v in (b.partition("=") for b in args.baseline)}

    R = Report()
    R.lines += ["# System audit", "", f"features: `{args.features}`"]
    check_health(ctx, R)
    check_coverage(ctx, R)
    check_predictive(ctx, R, baselines)
    check_explanations(ctx, R, args.explain_n, args.explain_k, args.seed)
    check_capabilities(ctx, R)
    R.check("NOT MEASURED", "extraction accuracy against hand labels",
            "run with --gold-sample 60, label the sheet, then --score-gold (the only way to measure it)")

    R.section("What is left (most urgent first)")
    todo = sorted((i for i in R.items if i[0] in PRIORITY), key=lambda i: PRIORITY[i[0]])
    for status, name, detail in todo:
        R.say(f"  {status:12} {name}")
    if not todo:
        R.say("  nothing flagged")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(R.lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()