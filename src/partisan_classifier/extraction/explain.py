"""What drives an article's score, and what is left when the opinion is removed.

Everything here works on a cached ArticleAnalysis, so no model has to run.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Callable

from . import subjects as S
from .subjects import ArticleAnalysis

OPINION_W = getattr(S, "OPINION_WEIGHT", 1.0)
FACT_W = getattr(S, "FACT_WEIGHT", getattr(S, "FACT_WIEGHT", 0.5))   # tolerates the old spelling


def reaggregate(a: ArticleAnalysis, keep: list[int]) -> ArticleAnalysis:
    """The analysis as if only the claims at indices `keep` existed.

    Subject stance is recomputed from each kept claim's stored (subject, score) targets, the same way
    analyze_article builds it, so features computed from the result reflect the smaller article.
    """
    claims = [a.claims[i] for i in keep]
    voice: dict[str, list] = defaultdict(list)
    quoted: dict[str, list] = defaultdict(list)
    by_lean: dict[tuple[str, str], list] = defaultdict(list)
    for info in claims:
        w = OPINION_W if info.claim.kind == "opinion" else FACT_W
        for key, score in info.targets:
            if info.claim.source is None:
                voice[key].append((score, w))
            else:
                quoted[key].append((score, w))
                by_lean[(key, info.source_lean or "unknown")].append((score, w))
    subjects = [
        replace(
            s,
            stance_voice=S._channel(voice.get(s.key, [])),
            stance_quoted=S._channel(quoted.get(s.key, [])),
            stance_by_source_lean={lean: ch for (k, lean), es in by_lean.items()
                                   if k == s.key and (ch := S._channel(es))},
        )
        for s in a.subjects
    ]
    return ArticleAnalysis(subjects, claims)


def claim_contributions(a: ArticleAnalysis, predict: Callable[[ArticleAnalysis], float]) -> list[float]:
    """Leave-one-out effect of each claim: predict(article) minus predict(article without that claim).

    Positive means the claim pushes the prediction up.
    """
    n = len(a.claims)
    base = predict(reaggregate(a, list(range(n))))
    return [base - predict(reaggregate(a, [j for j in range(n) if j != i])) for i in range(n)]


def ground_truth(a: ArticleAnalysis) -> dict:
    """The article with the opinion taken out: asserted facts, who said what, and the opinions as attributed.

    Editorial opinion (the article's own, unattributed) is dropped. Opinions quoted from a source are kept as
    statements by that source rather than as facts.
    """
    def text(c) -> str:
        return (c.claim.text_resolved or c.claim.text).strip()

    facts = [text(c) for c in a.claims if c.claim.source is None and c.claim.kind == "fact"]
    attributed_facts = [{"source": c.claim.source, "role": c.claim.source_role, "text": text(c)}
                        for c in a.claims if c.claim.source is not None and c.claim.kind == "fact"]
    statements = [{"source": c.claim.source, "role": c.claim.source_role, "speaker_type": c.speaker_type,
                   "type": c.opinion_type, "lean": c.source_lean, "text": text(c)}
                  for c in a.claims if c.claim.source is not None and c.claim.kind == "opinion"]
    dropped = sum(1 for c in a.claims if c.claim.source is None and c.claim.kind == "opinion")
    lines = list(facts) + [f"According to {f['source']}, {f['text']}" for f in attributed_facts]
    return {
        "subjects": [{"key": s.key, "names": s.names, "lean": s.political_lean} for s in a.primary()],
        "facts": facts,
        "attributed_facts": attributed_facts,
        "attributed_statements": statements,
        "editorial_opinions_removed": dropped,
        "text": "\n".join(lines),
    }