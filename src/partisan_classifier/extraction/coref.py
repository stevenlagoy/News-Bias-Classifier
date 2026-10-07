from __future__ import annotations

import argparse
import warnings
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Callable

warnings.filterwarnings("ignore", category=FutureWarning)

from . import claims as C
from .claims import Claim

# A cluster is a list of (start_char, end_char) mentions of one entity; exclusive
Cluster = list[tuple[int, int]]
Backend = Callable[[str, object], list[Cluster]]

NAME_LABELS = {"PERSON", "ORG", "NORP"}
# Pronoun forms are replaced by a name
# Plural forms are off by default: coreference models are unreliable with "they/their"
SINGULAR = {"he", "him", "his", "she", "her", "hers"}
PLURAL = {"they", "them", "their", "theirs"}
_RULE_PRONOUNS = SINGULAR
_SUBJ = {"nsubj", "nsubjpass"}


_FCOREF = None

def fastcoref_clusters(text: str, doc) -> list[Cluster]:
    global _FCOREF
    if _FCOREF is None:
        import torch
        import fastcoref.modeling as fm
        # transformers>=5 expects this attribute; fastcoref never calls post_init() to set it
        if not hasattr(fm.FCorefModel, "all_tied_weights_keys"):
            fm.FCorefModel.all_tied_weights_keys = {}
        from fastcoref import FCoref
        _FCOREF = FCoref(device="cuda:0" if torch.cuda.is_available() else "cpu")
    result = _FCOREF.predict(texts=[text])[0]
    return [[(a, b) for a, b in cl] for cl in result.get_clusters(as_strings=False)]


def _quote_mask(doc) -> list[bool]:
    mask = [False] * len(doc)
    in_quotation, open_sents = False, 0
    for sent in doc.sents:
        if in_quotation:
            open_sents += 1
            if open_sents > C.MAX_OPEN_SENTS:
                in_quotation, open_sents = False, 0
        for t in sent:
            if t.text == "“" or (t.text == '"' and not in_quotation):
                in_quotation = True
                continue
            if t.text == "”" or (t.text == '"' and in_quotation):
                in_quotation, open_sents = False, 0
                continue
            mask[t.i] = in_quotation
    return mask


def rule_clusters(text: str, doc) -> list[Cluster]:
    """Group person mentions by surname, then attach he/she pronouns to the best recent antecedent."""
    in_quotation = _quote_mask(doc)
    sent_of = {}
    for si, s in enumerate(doc.sents):
        for t in s:
            sent_of[t.i] = si
    spans: list[tuple[int, int]] = []
    covered: set[int] = set()
    for e in doc.ents:
        if e.label_ == "PERSON":
            spans.append((e.start, e.end))
            covered |= set(range(e.start, e.end))
    surnames = {doc[e - 1].lower_ for _, e in spans}
    for t in doc:
        if t.pos_ == "PROPN" and t.i not in covered and t.lower_ in surnames:
            spans.append((t.i, t.i + 1))

    def char_span(s: int, e: int) -> tuple[int, int]:
        return doc[s].idx, C._token_end(doc[e - 1])

    groups: dict[str, set[tuple[int, int]]] = defaultdict(set) # One of the data structures of all time
    for s, e in spans:
        groups[doc[e - 1].lower_].add(char_span(s, e))

    def score(s: int, e: int, dist: int) -> float:
        root = doc[e - 1]
        val = 1.0
        gov = None
        if root.dep_ in _SUBJ:
            val += 2
            gov = root.head
        elif root.dep_ == "appos" and root.head.dep_ in _SUBJ:
            val += 2
            gov = root.head.head
        if gov is not None and gov.dep_ == "ROOT":
            val += 1
        return val - 1.5 * dist

    for t in doc:
        if t.lower_ not in _RULE_PRONOUNS or in_quotation[t.i]:
            continue
        si = sent_of[t.i]
        best, best_key = None, None
        for s, e in spans:
            if e > t.i or in_quotation[s]:
                continue
            dist = si - sent_of[s]
            if dist < 0 or dist > 2:
                continue
            key = (score(s, e, dist), s)
            if best_key is None or key > best_key:
                best, best_key = (s, e), key
        if best is not None:
            groups[doc[best[1] - 1].lower_].add(char_span(t.i, t.i + 1))
    return [sorted(g) for g in groups.values() if len(g) > 1]

BACKENDS: dict[str, Backend] = {"fastcoref": fastcoref_clusters, "rules": rule_clusters}


def _canonical(doc, cluster: Cluster) -> str | None:
    """Fullest proper name among a cluster's mentions."""
    names: Counter[str] = Counter()
    for a, b in cluster:
        span = doc.char_span(a, b, alignment_mode="expand")
        if span is None or (len(span) == 1 and span[0].pos_ == "PRON"):
            continue
        ents = [e for e in span.ents if e.label_ in NAME_LABELS]
        if not ents:
            ents = [e for e in doc.ents if e.label_ in NAME_LABELS and e.start >= span.start and e.end <= span.end]
        if ents:
            name = max(ents, key=lambda e: len(e.text)).text
        elif all(t.pos_ == "PROPN" or t.is_punct for t in span):
            name = span.text
        else: continue
        name = name.removesuffix("'s").removesuffix("’s").strip()
        if name:
            names[name] += 1
    if not names:
        return None
    return max(names, key=lambda n: (len(n.split()), len(n), names[n]))


def _possessive(name: str) -> str:
    return name + ("’" if name.endswith("s") else "’s")


def _may_replace_source(span, name: str, kind: str) -> bool:
    """Only upgrade a source mention that is a pronoun, a partial name, or a person's role description."""
    if len(span) == 1 and span[0].lower_ in SINGULAR | PLURAL: return True
    if any(t.dep_ in {"cc", "conj"} for t in span): return False
    propn = {t.lower_ for t in span if t.pos_ == "PROPN"}
    name_words = {w.lower() for w in name.split()}
    if span.root.pos_ == "PROPN": return bool(propn) and propn <= name_words
    # common-noun head ("the US president", "the Louisiana lawmaker"): only role nouns, only for people
    return kind == "PERSON" and span.root.lemma_ in C.TITLES and not (propn & name_words)

def resolve_claims(article: str, claims: list[Claim], backend: str | Backend = "fastcoref", doc=None, rewrite_text: bool = True, plural: bool = False) -> list[Claim]:
    """Replace pronoun and partial-name sources with full names, for all claims of one article at once."""
    doc = doc if doc is not None else C._nlp(article)
    clusters = (BACKENDS[backend] if isinstance(backend, str) else backend)(article, doc)
    canon = [_canonical(doc, cl) for cl in clusters]
    pronouns = SINGULAR | PLURAL if plural else SINGULAR
 
    def find(a: int, b: int) -> int | None:
        best, best_ov = None, 0
        for ci, cl in enumerate(clusters):
            if canon[ci] is None:
                continue
            for s0, e0 in cl:
                ov = min(b, e0) - max(a, s0)
                if ov > best_ov:
                    best, best_ov = ci, ov
        return best
 
    out: list[Claim] = []
    canon = [c for c in canon if c is not None]
    for c in claims:
        new = replace(c)
        if c.source_start >= 0:
            ci = find(c.source_start, c.source_end)
            span = doc.char_span(c.source_start, c.source_end, alignment_mode="expand")
            if ci is not None and span is not None and _may_replace_source(span, *canon[ci]):
                new.source = canon[ci][0]
        edits: list[tuple[int, int, str]] = []
        for t in (doc if rewrite_text else []):
            if t.idx < c.start or C._token_end(t) > c.end or t.lower_ not in pronouns:
                continue
            ci = find(t.idx, C._token_end(t))
            if ci is None:
                continue
            name, kind = canon[ci]
            if t.lower_ in SINGULAR and kind != "PERSON":
                continue
            earlier = article[c.start : t.idx].lower()
            if name.split()[-1].lower() in earlier.replace("’s", " ").split():
                continue # the name is already in the claim; the pronoun is clear
            poss = t.tag_ == "PRP$" or t.lower_ in {"hers", "theirs"}
            edits.append((t.idx - c.start, C._token_end(t) - c.start, _possessive(name) if poss else name))
        text = article[c.start : c.end]
        for s0, e0, rep in sorted(edits, reverse=True):
            text = text[:s0] + rep + text[e0:]
        new.text_resolved = text
        out.append(new)
    return out

def process_articles(texts: list[str], backend: str | Backend = "fastcoref", rewrite_text: bool = True) -> list[list[Claim]]:
    """Extract and resolve claims for many articles. Batches classifier call."""
    per_article, docs = C.extract_many(texts, return_docs=True)
    return [resolve_claims(t, cs, backend, doc=d, rewrite_text=rewrite_text) for t, cs, d in zip(texts, per_article, docs)] # Honestly didn't know you could zip three lists like this


# Run like "python -m partisan_classifier.extraction.coref data/claims_samples.txt --backend fastcoref"
def main() -> None:
    ap = argparse.ArgumentParser(description="Extract claims and resolve pronoun sources.")
    ap.add_argument("path", help="UTF-8 text file; the whole file is treated as one article")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="fastcoref")
    args = ap.parse_args()
    with open(args.path, encoding="utf-8") as f:
        article = " ".join(ln.strip() for ln in f if ln.strip())
    for c in process_articles([article], args.backend)[0]:
        who = c.source or "-"
        if c.source_mention and c.source_mention != c.source:
            who = f"{c.source}  <- '{c.source_mention}'"
        print(f"[{c.kind:7}] ({who}) {c.text_resolved}")

if __name__ == "__main__":
    main()