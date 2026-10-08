from __future__ import annotations
from dataclasses import dataclass, field
import spacy
from collections import Counter
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

_nlp = spacy.load("en_core_web_sm") # en_core_web_trf is slower but probably more accurate
_classifier = None
LABELS = {"LABEL_0": "OBJ", "LABEL_1": "SUBJ"}

# Verbs which introduce someone's statement, matched by lemma
SPEECH = {
    "say", "tell", "argue", "claim", "warn", "announce", "state", "report",
    "add", "note", "suggest", "insist", "predict", "believe", "allege", "contend",
    "quip", "indicate", "find", "show", "write", "maintain", "assert", "acknowledge",
    "stress", "emphasize", "declare", "concede", "respond", "post", "confirm", "deny",
    "vow", "pledge", "conclude", "estimate",
}
# Speech verbs whose direct object can itself be the reported preposition (reflexive direct objectives)
REPORTING = {
    "find", "show", "report", "indicate", "suggest", "note", "estimate", "confirm",
    "conclude", "state", "announce", "predict", "reveal",
}
# Role nouns which precede a person's name
TITLES = {
    "leader", "president", "senator", "representative", "expert", "spokesman", "spokeswoman",
    "spokesperson", "governor", "mayor", "director", "chair", "chairman", "chairwoman", "minister",
    "secretary", "lawmaker", "official", "analyst", "professor", "commissioner", "candidate",
    "congressman", "congresswoman", "researcher", "economist",
}
VERBS = {"VERB", "AUX"}
SUBJ_DEPS = {"nsubj", "nsubjpass"}
COMP_DEPS = {"ccomp", "parataxis"}

MIN_QUOTE_CHARS = 15 # Detect longer sections of reported speech
MAX_OPEN_SENTS = 6   # Guard from unbalanced quotation marks 
_EDGE_PUNCT = {",", ";", ":", "-", "–", "—"}
_STRIP = " \t\n,;:-–—\"“”"


""" Reference for parts of speech symbols:
VERB = Verb
PROPN = Proper Name
PERSON = Personal Name
DATE = Date Noun
TIME = Time Noun
ROOT

nsubj
nsubjpass
ccomp
parataxis
cc
appos = Appositive
relcl
acl
compound
conj
dobj
obj

"""

@dataclass
class Claim:
    text: str
    source: str | None                  # Who a claim is attributed to (coref)
    kind: str                           # 'fact' or 'opinion'
    start: int = -1                     # character offset
    end: int = -1                       # character offset
    source_mention: str | None = None   # Source exactly as written in the article
    source_start: int = -1
    source_end: int = -1
    text_resolved: str | None = None    # text with pronouns replaced by names (coref)
    source_role: str | None = None      # how the article describes the source 

@dataclass
class _Src:
    text: str
    start: int
    end: int
    role: str | None = None # descriptor words around a person's name

@dataclass
class _Cand:
    text: str
    start: int
    end: int
    src: _Src | None

@dataclass
class _Att:
    src: _Src
    contents: list[tuple[int, int]]
    relaxed: bool = False
    residual: list[tuple[int, int]] = field(default_factory=list)


def _text(tokens) -> str:
    return "".join(t.text_with_ws for t in tokens).strip(" ,;")


def _token_end(t) -> int:
    return t.idx + len(t.text)


def _range(tokens):
    toks = list(tokens)
    return (toks[0].idx, _token_end(toks[-1])) if toks else None


def _trim_tokens(tokens, strip_cc: bool = False):
    toks = list(tokens)

    def junk(t) -> bool:
        return t.is_quote or t.text in _EDGE_PUNCT or (strip_cc and t.dep_ == "cc")

    while toks and (junk(toks[0]) or toks[0].lower_ == "that"):
        toks = toks[1:]
    while toks and junk(toks[-1]):
        toks = toks[:-1]
    return toks


def _quote_ranges(text: str, a0: int, a1: int, in_quote: bool) -> tuple[list[tuple[int, int]], bool]:
    """Character ranges of quoted text inside text[a0:a1], plus the quote state afterwards."""
    ranges: list[tuple[int, int]] = []
    start = a0 if in_quote else None
    for i in range(a0, a1):
        ch = text[i]
        if ch == "“" or (ch == '"' and not in_quote):
            in_quote, start = True, i + 1
        elif ch == "”" or (ch == '"' and in_quote):
            if start is not None and i > start:
                ranges.append((start, i))
            in_quote, start = False, None
    if in_quote and start is not None and a1 > start:
        ranges.append((start, a1))
    return ranges, in_quote


def _ent_at(doc, i):
    for e in doc.ents:
        if e.start <= i < e.end:
            return e
    return None


def _phrase(tok) -> _Src:
    """The noun phrase headed by tok, without appositives, relative clauses, or participles.""" # Really going back to middle school English class with this one
    skip: set[int] = set()
    for t in tok.subtree:
        if t is not tok and t.dep_ in {"appos", "relcl", "acl"}:
            skip |= {x.i for x in t.subtree}
    runs: list[list] = []
    for t in tok.subtree:
        if t.i in skip: continue
        if runs and runs[-1][-1].i == t.i - 1:
            runs[-1].append(t)
        else:
            runs.append([t])
    run = next((r for r in runs if any(t.i == tok.i for t in r)), [tok])
    kept = _trim_tokens(run, strip_cc=True) or [tok]
    txt = _text(kept)
    return _Src(txt, kept[0].idx, _token_end(kept[-1]), txt)


def _role_text(subj, exclude: set[int]) -> str | None:
    """Words describing a named person: titles and appositives."""
    skip: set[int] = set()
    for t in subj.subtree:
        if t is not subj and t.dep_ in {"relcl", "acl"}:
            skip |= {x.i for x in t.subtree}
    tokens = []
    for t in subj.subtree:
        if t.is_quote or t.text == ":": break # Stop where reported speech begins
        if t.i not in skip and t.i not in exclude:
            tokens.append(t)
    tokens = _trim_tokens(tokens, strip_cc=True)
    if all(t.pos_ == "DET" for t in tokens): return None
    return _text(tokens) or None


def _propn_run(token) -> list:
    """The adjacent run of proper-noun tokens around token."""
    doc = token.doc
    low = high = token.i
    while low > 0 and doc[low - 1].pos_ == "PROPN":
        low -= 1
    while high + 1 < len(doc) and doc[high + 1].pos_ == "PROPN":
        high += 1
    return list(doc[low : high + 1])


def _name_span(subj) -> _Src:
    """Best name for a subject: a PERSON entity in the subject or its appositives, else the phrase."""
    doc = subj.doc
    heads, stack = [subj], [subj]
    while stack:
        for c in stack.pop().children:
            if c.dep_ == "appos":
                heads.append(c)
                stack.append(c)
    for t in heads:
        if t.ent_type_ == "PERSON":
            e = _ent_at(doc, t.i)
            if e is not None:
                return _Src(e.text, e.start_char, e.end_char, _role_text(subj, set(range(e.start, e.end))))
    if subj.lemma_ in TITLES:
        for c in subj.children:
            if c.dep_ == "compound" and c.ent_type_ == "PERSON":
                e = _ent_at(doc, c.i)
                if e is not None:
                    return _Src(e.text, e.start_char, e.end_char, _role_text(subj, set(range(e.start, e.end))))
    for c in subj.children:
        if c.dep_ == "appos" and c.pos_ == "PROPN":
            return _phrase(c)
    if subj.pos_ == "PROPN" and not any(c.dep_ in {"conj", "cc"} for c in subj.children):
        # name was NER mislabeled as ORG
        run = _propn_run(subj)
        txt = _text(run)
        return _Src(txt, run[0].idx, _token_end(run[-1]), _role_text(subj, {t.i for t in run}))
    return _phrase(subj)


def _is_speech(tok) -> bool:
    if tok.pos_ != "VERB": return False
    if tok.lemma_ in SPEECH: return True
    return tok.lemma_ == "go" and any(c.lower_ == "on" for c in tok.children) # "X went on: ..."


def _subject(tok, in_quote):
    """The speaker's token. Ignores subjects inside quotations and resolves relative pronouns."""
    def ok(c): return c.dep_ == "nsubj" and not in_quote(c)

    def closest(cs): return min(cs, key=lambda c: abs(c.i - tok.i))

    candidates = [c for c in tok.children if ok(c)]
    if candidates:
        s = closest(candidates)
        if s.lower_ in {"who", "which", "that"} and tok.dep_ == "relcl":
            return tok.head
        return s
    if tok.dep_ in {"acl", "relcl"} and tok.head.pos_ in {"NOUN", "PROPN"} and not in_quote(tok.head):
        return tok.head
    if tok.dep_ in {"advcl", "conj", "xcomp"}:
        for a in tok.ancestors:
            if a.pos_ in VERBS:
                cs = [c for c in a.children if ok(c)]
                if cs:
                    return closest(cs)
    return None


def _window_source(tok, sent, in_quote) -> _Src | None:
    """Fallback when parse gives no subject: look at words next to the verb."""
    doc = tok.doc
    left, i = [], tok.i - 1
    while i >= sent.start and not in_quote(doc[i]) and not doc[i].is_quote:
        left.append(doc[i])
        i -= 1
    left.reverse()
    low = left[0].i if left else tok.i
    high = left[-1].i + 1 if left else tok.i
    for e in doc.ents:
        if e.label_ == "PERSON" and e.start >= low and e.end <= high:
            return _Src(e.text, e.start_char, e.end_char)
    right = tok.i + 1
    for nc in doc.noun_chunks:
        if nc.start == right and not in_quote(doc[right]) and nc.root.ent_type_ not in {"DATE", "TIME"}:
            return _phrase(nc.root)
    for nc in doc.noun_chunks:
        if left and nc.start >= low and nc.end <= high:
            return _phrase(nc.root)
    return None


def _groups(sent, exclude: set[int], skip_ranges):
    """Contiguous runs of sentence tokens outside `exclude` and outside the given character ranges."""
    out, cur = [], []
    for t in sent:
        if t.i in exclude or any(a <= t.idx < b for a, b in skip_ranges):
            if cur:
                out.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    ranges = []
    for g in out:
        r = _range(_trim_tokens(g, strip_cc=True))
        if r: ranges.append(r)
    return ranges


def _contents(verb, subj, src, sent, qr):
    """Where the reported content sits, as (character ranges, relaxed-gate flag), or None."""
    doc = verb.doc
    dobj = any(c.dep_ in {"dobj", "obj"} for c in verb.children)
    region = list(doc[verb.i + 1 : verb.right_edge.i + 1])
    inverted = src.start > verb.idx # "..., said X."
    comps = [c for c in verb.children if c.dep_ in {"ccomp", "parataxis"}]
    after = [c for c in comps if c.i > verb.i and c.dep_ == "ccomp"]
    before = [c for c in comps if c.i < verb.i]
    if inverted: # Clause which swallowed speaker is a misparse
        after = [c for c in after if not any(src.start <= t.idx < src.end for t in c.subtree)]

    def ranges_of(cs):
        rs = [_range(_trim_tokens(c.subtree)) for c in cs]
        return [r for r in rs if r]

    if after and not inverted:
        covered = sum(len(list(c.subtree)) for c in after)
        if verb.dep_ in {"ROOT", "xcomp"} and verb.lemma_ in REPORTING and dobj and covered < 0.6 * len(region):
            r = _range(_trim_tokens(region)) # Parser split the clause badly; take everything after the verb
            return ([r], True) if r else None
        rs = ranges_of(after)
        return (rs, False) if rs else None

    long_q = [(a, b) for a, b in qr if b - a >= MIN_QUOTE_CHARS]
    if long_q:
        return long_q, False

    rs = ranges_of(before + after)
    if rs: return rs, False

    prev = doc[subj.left_edge.i - 1] if subj is not None and subj.left_edge.i > sent.start else None
    bare = not any(c.dep_ in {"dobj", "obj", "ccomp", "xcomp", "attr"} for c in verb.children)
    if (subj is not None and subj.i < verb.i and prev is not None and prev.text in {",", "–", "—", ":"} and bare and verb.right_edge.i >= sent.end - 2):
        pre = list(doc[sent.start : subj.left_edge.i]) # "..., the Reuters/Ipsos poll found."
        if len(pre) >= 5:
            r = _range(_trim_tokens(pre))
            return ([r], False) if r else None

    if verb.dep_ in {"ROOT", "xcomp"} and verb.lemma_ in REPORTING and dobj:
        r = _range(_trim_tokens(region))
        return ([r], True) if r else None
    return None


def _according(sent, in_quote) -> _Att | None:
    doc = sent.doc
    for t in sent:
        if t.lower_ == "according" and t.i + 1 < sent.end and doc[t.i + 1].lower_ == "to" and not in_quote(t):
            pobj = next((c for c in doc[t.i + 1].children if c.dep_ == "pobj"), None)
            if pobj is None: continue
            ranges = _groups(sent, {x.i for x in t.subtree}, [])
            if ranges: return _Att(_name_span(pobj), ranges)
    return None


def _find_attribution(sent, qr, in_quote):
    best = None
    for tok in sent:
        if in_quote(tok) or not _is_speech(tok): continue
        subj = _subject(tok, in_quote)
        src = _name_span(subj) if subj is not None else _window_source(tok, sent, in_quote)
        if src is None: continue
        got = _contents(tok, subj, src, sent, qr)
        if got is None: continue
        residual = []
        if tok.dep_ not in {"ROOT", "xcomp"}: # Rest of sentence is another claim
            residual = _groups(sent, {t.i for t in tok.subtree}, got[0])
        att = _Att(src, got[0], got[1], residual)
        if tok.dep_ == "ROOT": return att
        best = best or att
    return best or _according(sent, in_quote)


def _split_conj(span):
    """Split conjunctions, like 'A did X and B did Y', as long as the second verb has its own subject."""
    for token in span:
        if (token.dep_ == "conj" and token.pos_ in VERBS and token.head.pos_ in VERBS and any(c.dep_ in SUBJ_DEPS for c in token.children)):
            sub = span.doc[token.left_edge.i : token.right_edge.i + 1]
            drop = {t.i for t in sub}
            drop |= {t.i for t in token.head.children if t.dep_ == "cc" and t.i < token.i}
            rest = [t for t in span if t.i not in drop]
            if rest:
                yield span.doc[rest[0].i : rest[-1].i + 1]
            yield sub
            return
    yield span


def _is_claim(span, relaxed: bool = False) -> bool:
    if len(span) < 5 or span.text.strip().endswith("?"):
        return False
    if not any(t.pos_ in VERBS for t in span):
        return False
    return relaxed or any(t.dep_ in SUBJ_DEPS for t in span)


def _trim_chars(text: str, a: int, b: int):
    while a < b and text[a] in _STRIP:
        a += 1
    while b > a and text[b - 1] in _STRIP:
        b -= 1
    seg = text[a:b]
    if seg.count('"') % 2 == 1:   # straight quotes: pull in the partner quote just outside the claim
        if b < len(text) and text[b] == '"':
            b += 1
        elif a > 0 and text[a - 1] == '"':
            a -= 1
    elif seg.count("“") > seg.count("”"):
        j = b
        while j < len(text) and j - b < 4 and text[j] in ".!?,”\"":
            j += 1
            if text[j - 1] in "”\"":
                b = j
                break
    elif seg.count("”") > seg.count("“") and a > 0 and text[a - 1] in "“\"":
        a -= 1
    return a, b


def _claims_in(text: str, ranges, relaxed: bool, src: _Src | None) -> list[_Cand]:
    out: list[_Cand] = []
    for a, b in ranges:
        seg = text[a:b]
        if not seg.strip(): continue
        # Re-parse the isolated text so the gate and splitter see clean clauses
        for sent in _nlp(seg).sents:
            for part in _split_conj(sent):
                if _is_claim(part, relaxed):
                    s0, e0 = _trim_chars(text, a + part[0].idx, a + part[-1].idx + len(part[-1].text))
                    if e0 > s0: out.append(_Cand(text[s0:e0], s0, e0, src))
    return out


def _candidates(doc) -> list[_Cand]:
    text = doc.text
    candidates: list[_Cand] = []
    in_quote = False
    quote_src: _Src | None = None
    pending: list[_Cand] = [] # candidates from current multi-sentence quotation
    open_sents = 0

    for sent in doc.sents:
        was_open = in_quote
        qr, in_quote = _quote_ranges(text, sent.start_char, sent.end_char, in_quote)

        def inq(t, _r=qr): return any(a <= t.idx < b for a, b in _r)

        att = _find_attribution(sent, qr, inq)
        if att:
            new = _claims_in(text, att.contents, att.relaxed, att.src)
            new += _claims_in(text, att.residual, False, None)
            new.sort(key=lambda c: c.start)
        else:
            effective = quote_src if was_open else None
            new = _claims_in(text, [(sent.start_char, sent.end_char)], False, effective)
        candidates.extend(new)

        if in_quote or was_open:
            if not was_open: # This sentence opens the quotation
                quote_src, pending, open_sents = (att.src if att else None), [], 0

            pending.extend(new)
            open_sents += 1
            if not in_quote and was_open:
                # CLosing sentence usually carries attribution; backfill earlier parts
                final = (att.src if att else None) or quote_src
                for c in pending:
                    if c.src is None:
                        c.src = final
            if not in_quote or open_sents > MAX_OPEN_SENTS:
                in_quote, quote_src, pending, open_sents = False, None, [], 0
    return candidates


def _classify(texts: list[str]) -> list[str]:
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        _classifier = pipeline(
            "text-classification",
            model="GroNLP/mdebertav3-subjectivity-multilingual",
            truncation=True,
        )
    preds = _classifier(texts, batch_size=32)
    return ["fact" if LABELS[p["label"]] == "OBJ" else "opinion" for p in preds]


def _to_claims(candidates: list[_Cand], kinds: list[str]) -> list[Claim]:
    return [
        Claim(
            text=c.text, source=c.src.text if c.src else None, kind=k,
            start=c.start, end=c.end,
            source_mention=c.src.text if c.src else None,
            source_start=c.src.start if c.src else -1,
            source_end=c.src.end if c.src else -1,
            source_role=c.src.role if c.src else None
        )
        for c, k in zip(candidates, kinds)
    ]


def extract_many(texts: list[str], return_docs: bool = False):
    """Extract claims from many articles with one batched classifier call."""
    docs = list(_nlp.pipe(texts, batch_size=8))
    per_doc = [_candidates(d) for d in docs]
    flat = [c for cs in per_doc for c in cs]
    kinds = _classify([c.text for c in flat]) if flat else []
    out, i = [], 0
    for cs in per_doc:
        out.append(_to_claims(cs, kinds[i : i + len(cs)]))
        i += len(cs)
    return (out, docs) if return_docs else out


def extract_claims(text: str) -> list[Claim]:
    return extract_many([text])[0]


def _audit(texts):
    counts = Counter()
    for doc in _nlp.pipe(texts, batch_size=64):
        for token in doc:
            if token.dep_ == "ccomp" and token.head.pos_ == "VERB" and token.head.lemma_ not in SPEECH:
                counts[token.head.lemma_] += 1
    return counts.most_common(50)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract claims from text.")
    ap.add_argument("path", help="UTF-8 text file, one passage per line")
    ap.add_argument("--joined", action="store_true", help="treat the whole file as a single article")
    ap.add_argument("--audit", action="store_true", help="list ccomp-head verbs missing from SPEECH")
    args = ap.parse_args()
 
    with open(args.path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    passages = [" ".join(lines)] if args.joined else lines
    if args.audit:
        print(_audit(passages))
    for claims in extract_many(passages):
        for c in claims:
            print(f"[{c.kind:7}] ({c.source or '-'}) {c.text}") # type: ignore
        print()


if __name__ == "__main__":
    main()