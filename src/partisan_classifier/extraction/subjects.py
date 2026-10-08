from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Callable

from . import claims as C
from . import coref as R
from .subjects_politics import SUBJECTS_ALIGNMENTS
from .claims import Claim


SUBJ_DEPS = {"nsubj", "nsubjpass"}
MODIFIER_DEPS = {"amod", "compound", "nmod"}
NAME_LABELS = {"PERSON", "ORG", "NORP"}
MODIFIER_WEIGHT = 0.25  # a name used as a modifier doesn't say much about the subject
PRONOUN_WEIGHT = 0.5
LEAD_SENTENCES = 2      # mentions in the first few sentences are more impactful
LEAD_BONUS = 2.0        # mentions in the first few sentences are more impactful
TOP_K = 6               # at most this many primary subjects
MIN_PRIMARY = 3.0       # absolute floor on subject score
REL_PRIMARY = 0.3       # relative to the top-scoring subject
STANCE_BAND = 0.2       # |score| above this counts as positive / negative, |score| less than this is neutral
OPINION_WEIGHT = 1.0    # opinon claims weigh this much for stance
FACT_WEIGHT = 0.5       # factual claims weigh this much for stance

StanceFn = Callable[[list[tuple[str, str]]], list[float]] # [(text, target)] -> scores in [-1, 1]


_TITLE = re.compile(
    r"^(?:the\s+)?(?:(?:vice[- ]president|president|sen\.|senator|rep\.|representative|gov\.|governor|mayor|dr\.|secretary)\s+)+",
    re.I,
)


def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    for suffix in ("’s", "'s"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    s = _TITLE.sub("", s)
    return re.sub(r"^the\s+", "", s, flags=re.I).casefold()


class AliasIndex:
    """Fact, case- and title-insensitive lookup over subject table."""

    def __init__(self, table: dict):
        self.table = table
        self._norm_to_key: dict[str, str] = {}
        raw: set[str] = set()
        for key, entry in table.items():
            for name in [key, *entry["names"]]:
                self._norm_to_key.setdefault(_norm(name), key)
                raw.add(name)
        pats = sorted(raw, key=len, reverse=True)
        self.regex = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(p) for p in pats) + r")(?!\w)", re.I) if pats else None

    def key_of(self, text: str) -> str | None:
        return self._norm_to_key.get(_norm(text))

    def lean(self, key: str | None) -> str | None:
        entry = self.table.get(key) if key else None
        return str(entry["political_lean"]) if entry and entry.get("political_lean") else None


@dataclass
class Mention:
    start: int
    end: int
    text: str
    key: str
    kind: str # "alias" | "ner" | "pronoun" | "descriptor"
    weight: float
    sent: int
    is_subject: bool = False
    person: bool = False

@dataclass
class ChannelStance:
    n: int
    pos: int
    neu: int
    neg: int
    mean: float
    label: str

@dataclass
class Subject:
    key: str
    names: list[str]
    political_lean: str | None
    is_person: bool
    mentions: float
    sentences: int
    as_subject: int
    speaker_claims: int
    score: float
    primary: bool = False
    stance_voice: ChannelStance | None = None   # article's own voice
    stance_quoted: ChannelStance | None = None  # what quoted sources say about the subject
    stance_by_source_lean: dict[str, ChannelStance] = field(default_factory=dict)

@dataclass
class ClaimInfo:
    claim: Claim
    speaker_type: str   # article | expert | official | data | anonymous | layperson | unknown
    opinion_type: str   # objective fact | editorial opinion | expert opinion/prediction | ...
    first_person: bool
    source_key: str | None
    source_lean: str | None
    source_lean_basis: str | None # "table" or "role"
    targets: list[tuple[str, float]] = field(default_factory=list) # (subject key, stance score)

@dataclass
class ArticleAnalysis:
    subjects: list[Subject]
    claims: list[ClaimInfo]

    def primary(self) -> list[Subject]:
        return [s for s in self.subjects if s.primary]

    def to_dict(self) -> dict:
        return asdict(self)


def _weight(span) -> float:
    return MODIFIER_WEIGHT if span.root.dep_ in MODIFIER_DEPS else 1.0


def _is_subject(root) -> bool:
    return root.dep_ in SUBJ_DEPS or (root.dep_ == "appos" and root.head.dep_ in SUBJ_DEPS)


def _looks_like_name(text: str) -> bool:
    tokens = text.replace("’s", "").split()
    return 1 <= len(tokens) <= 4 and all(t[:1].isupper() and not any(ch.isdigit() for ch in t) for t in tokens)


def find_mentions(
    doc, index: AliasIndex, clusters: list[R.Cluster] | None = None,
    claims: list[Claim] | None = None
) -> list[Mention]:
    sent_of = {t.i: si for si, s in enumerate(doc.sents) for t in s}
    mentions: list[Mention] = []
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return not any(a < e and s < b for s, e in taken)

    def add(a, b, key, kind, w, span, person=False):
        mentions.append(Mention(
            a, b, doc.text[a:b], key, kind, w, sent_of.get(span.start, 0), _is_subject(span.root), person
        ))
        taken.append((a, b))

    # Aliases from the table
    if index.regex:
        for m in index.regex.finditer(doc.text):
            span = doc.char_span(m.start(), m.end(), alignment_mode="expand")
            key = index.key_of(m.group())
            if span is not None and key is not None:
                add(m.start(), m.end(), key, "alias", _weight(span), span, any(t.ent_type_ == "PERSON" for t in span))

    # Other named people, groups, institutions from NER
    people = [e for e in doc.ents if e.label_ == "PERSON" and free(e.start_char, e.end_char)]
    fullest: dict[str, str] = {}
    for e in people:
        last = e[-1].lower_
        if len(e.text) > len(fullest.get(last, "")):
            fullest[last] = e.text
    for e in people:
        add(e.start_char, e.end_char, index.key_of(fullest[e[-1].lower_]) or fullest[e[-1].lower_], "ner", _weight(e), e, True)
    for t in doc: # Base surnames missed by NER
        if t.pos_ == "PROPN" and t.lower_ in fullest and free(t.idx, C._token_end(t)):
            add(t.idx, C._token_end(t), index.key_of(fullest[t.lower_]) or fullest[t.lower_], "ner", _weight(doc[t.i : t.i + 1]), doc[t.i : t.i + 1], True)
    orgs: dict[str, str] = {}
    for e in doc.ents:
        if e.label_ in {"ORG", "NORP"} and free(e.start_char, e.end_char):
            w = _weight(e)
            if e.label_ == "NORP" and w < 1.0: continue # Adjective, not a group mention
            display = index.key_of(e.text) or orgs.setdefault(_norm(e.text), re.sub(r"^[Tt]he\s+", "", e.text))
            add(e.start_char, e.end_char, display, "ner", w, e)

    # Pronouns and role descriptions (coreferences with known subject)
    person_keys = {m.key for m in mentions if m.person}
    for cl in clusters or []:
        votes: Counter[str] = Counter()
        for a, b in cl:
            for m in mentions:
                if m.kind in {"alias", "ner"} and a < m.end and m.start < b:
                    votes[m.key] += 1
        if not votes: continue
        spans = [(a, b, doc.char_span(a, b, alignment_mode="expand")) for a, b in cl]
        has_pronoun = any(sp is not None and len(sp) == 1 and sp[0].lower_ in R.SINGULAR for _, _, sp in spans)
        pool = [k for k in votes if k in person_keys] if has_pronoun else list(votes)
        if not pool: continue
        main = max(pool, key=lambda k: votes[k])
        for a, b, sp in spans:
            if sp is None or not free(a, b): continue
            if len(sp) == 1 and sp[0].lower_ in R.SINGULAR:
                if main in person_keys:
                    add(a, b, main, "pronoun", PRONOUN_WEIGHT, sp, main in person_keys)
            elif R._may_replace_source(sp, main, "PERSON" if main in person_keys else "ORG"):
                add(a, b, main, "descriptor", PRONOUN_WEIGHT, sp, main in person_keys)

    # Claim sources that look like names but are mislabeled or missed by NER
    names = {c.source for c in claims or [] if c.source and _looks_like_name(c.source)}
    longest: dict[str, str] = dict(fullest)
    for n in names:
        last = n.split()[-1].lower()
        if len(n) > len(longest.get(last, "")):
            longest[last] = n
    for c in claims or []:
        if c.source in names and c.source_start >= 0 and free(c.source_start, c.source_end):
            full = longest[c.source.split()[-1].lower()]
            span = doc.char_span(c.source_start, c.source_end, alignment_mode="expand")
            if span is not None:
                add(c.source_start, c.source_end, index.key_of(full) or full, "ner", _weight(span), span, True)
    
    return sorted(mentions, key=lambda m: m.start)


_ABSA = None # Aspect-based Sentiment Analysis


def absa_stance(pairs: list[tuple[str, str]]) -> list[float]:
    """Aspect-based sentiment: how does `text` portray `target`? Returns P(positive) - P(negative)."""
    global _ABSA
    if not pairs: return []
    if _ABSA is None:
        from transformers import pipeline
        _ABSA = pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1", top_k=None, truncation=True)
    outs = _ABSA([{"text": t, "text_pair": a} for t, a in pairs], batch_size=16)
    scores = []
    for o in outs:
        p = {d["label"].lower(): d["score"] for d in o} # type: ignore
        scores.append(p.get("positive", 0.0) - p.get("negative", 0.0)) # type: ignore
    return scores


def _channel(entries: list[tuple[float, float]]) -> ChannelStance | None:
    """entries: (score, weight) pairs"""
    if not entries: return None
    total = sum(w for _, w in entries)
    mean = sum(s * w for s, w in entries) / total if total else 0.0
    pos = sum(1 for s, _ in entries if s > STANCE_BAND)
    neg = sum(1 for s, _ in entries if s < -STANCE_BAND)
    label = "positive" if mean > STANCE_BAND else "negative" if mean < -STANCE_BAND else "neutral"
    return ChannelStance(len(entries), pos, len(entries) - pos - neg, neg, round(mean, 3), label)


def _words(text: str) -> set[str]:
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z]+", text.lower())}


EXPERT = {
    "expert", "professor", "analyst", "economist", "researcher", "scientist", "scholar", "historian",
    "pollster", "statistician", "physician", "doctor", "fellow", "academic", "specialist", "research",
    "university", "institute", "institution", "laboratory"
}
DATA = {
    "poll", "survey", "study", "report", "analysis", "data", "measure", "rating", "audit"
}
OFFICIAL = {
    "senator", "governor", "mayor", "president", "minister", "secretary", "representative", "congressman",
    "congresswoman", "lawmaker", "chair", "chairman", "chairwoman", "spokesman", "spokeswoman",
    "spokesperson", "communication", "commissioner", "candidate", "leader", "speaker", "supervisor",
    "house", "administration", "department", "agency", "head", "congress", "senate", "attorney"
}
MEDIA = {"outlet", "newspaper", "news", "reporter", "journalist", "network"}
ANON = {"person", "people", "source"}
ANON_CUES = {"familiar", "knowledge", "anonymity", "condition", "unnamed"}
LAY = {"supporter", "voter", "resident", "protester", "citizen"}
LEFT_HINTS = {"democratic", "democrat", "progressive", "liberal"}
RIGHT_HINTS = {"republican", "gop", "conservative"}
_FIRST_PERSON = re.compile(r"\b(?:I|I’m|I'm|I’ve|I've|I’ll|I'll|my|me|mine)\b")


def classify_speaker(c: Claim, in_table: bool, extra_role: str = "") -> str:
    if not c.source: return "article"
    text = f"{c.source_role or ''} {extra_role} {c.source or ''}"
    w = _words(text)
    plain = (c.source_role or c.source).strip().lower()
    if w & ANON and (w & ANON_CUES or plain in {"the person", "a person", "the source", "a source"}):
        return "anonymous"
    if w & DATA:
        return "data"
    if w & EXPERT:
        return "expert"
    if w & MEDIA:
        return "media"
    if w & OFFICIAL:
        return "official"
    if w & LAY:
        return "layperson"
    return "official" if in_table else "unknown"


def classify_claim(c: Claim, speaker: str) -> str:
    if not c.source:
        return "objective fact" if c.kind == "fact" else "editorial opinion"
    if speaker == "data":
        return "data finding"
    if speaker == "expert":
        return "expert opinion" if c.kind == "opinion" else "expert statement"
    return "personal opinion" if c.kind == "opinion" else "attributed fact"


def source_lean(c: Claim, source_key: str | None, index: AliasIndex, extra_role: str = "") -> tuple[str | None, str | None]:
    lean = index.lean(source_key)
    if lean: return lean, "table"
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", f"{c.source_role or ''} {extra_role} {c.source or ''}")}
    words = {w[:-1] if w.endswith("s") else w for w in words}
    left, right = bool(words & LEFT_HINTS), bool(words & RIGHT_HINTS)
    if left != right:
        return ("left" if left else "right"), "role"
    return None, None


def person_roles(doc, mentions: list[Mention]) -> dict[str, str]:
    """Descriptions of each person whereve the article gives them."""
    roles: dict[str, list[str]] = defaultdict(list)
    people = {m.key for m in mentions if m.person} # NER sometimes labels a person ORG
    for m in mentions:
        if m.kind not in {"alias", "ner"} or m.key not in people: continue
        span = doc.char_span(m.start, m.end, alignment_mode="expand")
        if span is None: continue
        head = span.root
        pre = [t.text for t in head.lefts if t.dep_ in {"compound", "amod"} and t.i < span.start]
        post = []
        for c in head.children:
            if c.dep_ == "appos":
                skip = {x.i for t in c.subtree if t.dep_ in {"relcl", "acl"} for x in t.subtree}
                post.append(C._text([t for t in c.subtree if t.i not in skip]))
        for txt in (" ".join(pre), *post):
            if txt and txt not in roles[m.key]:
                roles[m.key].append(txt)
    return {k: "; ".join(v[:3]) for k, v in roles.items()}


def analyze_article(
    article: str, claims: list[Claim], doc, clusters: list[R.Cluster] | None = None,
    index: AliasIndex | None = None, stance_fn: StanceFn | None = absa_stance
) -> ArticleAnalysis:
    index = index or AliasIndex(SUBJECTS_ALIGNMENTS)
    mentions = find_mentions(doc, index, clusters, claims)

    # Which subject is speaking per claim
    source_key: list[str | None] = []
    for c in claims:
        key = None
        if c.source_start >= 0:
            hit = next((m for m in mentions if m.start < c.source_end and c.source_start < m.end), None)
            key = hit.key if hit else None
        source_key.append(key or (index.key_of(c.source) if c.source else None))

    # Subjects
    by_key: dict[str, list[Mention]] = defaultdict(list)
    for m in mentions: by_key[m.key].append(m)
    speaker_claims = Counter(k for k in source_key if k)
    subjects: dict[str, Subject] = {}
    for key, ms in by_key.items():
        weighted = sum(m.weight for m in ms)
        as_subject = sum(1 for m in ms if m.is_subject and m.kind != "pronoun")
        lead = any(m.sent < LEAD_SENTENCES and m.kind in {"alias", "ner"} for m in ms)
        surface = Counter(re.sub(r"[’']s$", "", m.text) for m in ms if m.kind in {"alias", "ner"})
        subjects[key] = Subject(
            key=key, names=[n for n, _ in surface.most_common(5)], political_lean=index.lean(key),
            is_person=any(m.person for m in ms), mentions=round(weighted, 2),
            sentences=len({m.sent for m in ms}), as_subject=as_subject, speaker_claims=speaker_claims[key],
            score=round(weighted + 0.5 * as_subject + 0.5 * speaker_claims[key] + (LEAD_BONUS if lead else 0), 2),
        )
    ranked = sorted(subjects.values(), key=lambda s: -s.score)
    if ranked:
        floor = max(MIN_PRIMARY, REL_PRIMARY * ranked[0].score)
        for s in ranked[:TOP_K]:
            s.primary = s.score >= floor
    # Claim typing
    roles = person_roles(doc, mentions)
    infos: list[ClaimInfo] = []
    for c, sk in zip(claims, source_key):
        extra = roles.get(sk, "") if sk else ""
        speaker = classify_speaker(c, sk in index.table if sk else False, extra)
        lean, basis = source_lean(c, sk, index, extra)
        infos.append(ClaimInfo(c, speaker, classify_claim(c, speaker), bool(_FIRST_PERSON.search(c.text)), sk, lean, basis))

    # Stance: one (claim text, target) job per subject mentioned in a claim
    if stance_fn is not None:
        jobs: list[tuple[int, str, tuple[str, str]]] = []
        for ci, (c, sk) in enumerate(zip(claims, source_key)):
            inside: dict[str, list[Mention]] = defaultdict(list)
            for m in mentions:
                if c.start <= m.start and m.end <= c.end:
                    inside[m.key].append(m)
            for key, ms in inside.items():
                if key == sk: continue # Speaker's self-presentation (not a stance)
                surface = next((m.text for m in ms if m.kind in {"alias", "ner"}), None)
                if surface: jobs.append((ci, key, (c.text, surface)))
                elif c.text_resolved and key.split()[-1].lower() in c.text_resolved.lower():
                    jobs.append((ci, key, (c.text_resolved, key)))
        scores = stance_fn([p for _, _, p in jobs])
        voice: dict[str, list] = defaultdict(list)
        quoted: dict[str, list] = defaultdict(list)
        by_lean: dict[tuple[str, str], list] = defaultdict(list)
        for (ci, key, _), sc in zip(jobs, scores):
            c, info = claims[ci], infos[ci]
            info.targets.append((key, round(sc, 3)))
            entry = (sc, OPINION_WEIGHT if c.kind == "opinion" else FACT_WEIGHT)
            if c.source is None:
                voice[key].append(entry)
            else:
                quoted[key].append(entry)
                by_lean[(key, info.source_lean or "unknown")].append(entry)
        for key, s in subjects.items():
            s.stance_voice = _channel(voice.get(key, []))
            s.stance_quoted = _channel(quoted.get(key, []))
            s.stance_by_source_lean = {
                lean: ch for (k, lean), es in by_lean.items() if k == key and (ch := _channel(es))
            }
    return ArticleAnalysis(ranked, infos)



def analyze_articles(
    texts: list[str], backend: str | R.Backend = "fastcoref",
    stance_fn: StanceFn | None = absa_stance, table: dict | None = None
) -> list[ArticleAnalysis]:
    """Full pipeline for many articles: claims, one coreference pass, subjects, stance, claim typing."""
    index = AliasIndex(table if table is not None else SUBJECTS_ALIGNMENTS)
    per_article, docs = C.extract_many(texts, return_docs=True)
    out = []
    for text, claims, doc in zip(texts, per_article, docs):
        clusters = (R.BACKENDS[backend] if isinstance(backend, str) else backend)(text, doc)
        claims = R.resolve_claims(text, claims, backend, doc=doc, clusters=clusters)
        out.append(analyze_article(text, claims, doc, clusters, index, stance_fn))
    return out


def _fmt(ch: ChannelStance | None) -> str:
    return "-" if ch is None else f"{ch.label} ({ch.mean:+.2f}, n={ch.n})"


"""
Take an article. Chunk it as necessary.
Identify all subjects (people, groups, institutions).
Identify each subject's alignment from subjects_politics lookup table.
Identify what sentences and claims are associated with each subject.
Determine the valence of those sentences and claims: positive, neutral, or negative presentation of the subject.
Build output with all per-subject information.
"""
def main() -> None:
    ap = argparse.ArgumentParser(description="Primary subjects, stance, and statement types for one article.")
    ap.add_argument("path", help="UTF-8 text file; the whole file is treated as one article")
    ap.add_argument("--backend", choices=sorted(R.BACKENDS), default="fastcoref")
    ap.add_argument("--no-stance", action="store_true", help="skip the stance model")
    args = ap.parse_args()
    with open(args.path, encoding="utf-8") as f:
        article = " ".join(ln.strip() for ln in f if ln.strip())
    result = analyze_articles([article], args.backend, None if args.no_stance else absa_stance)[0]
 
    print("PRIMARY SUBJECTS")
    for s in result.primary():
        print(f"  {s.key:28} lean={s.political_lean or '?':6} score={s.score:5.1f}  "
              f"voice: {_fmt(s.stance_voice)}  quoted: {_fmt(s.stance_quoted)}")
        for lean, ch in s.stance_by_source_lean.items():
            print(f"      said by {lean:8} {_fmt(ch)}")
    print("\nATTRIBUTED CLAIMS")
    for ci in result.claims:
        if ci.claim.source:
            lean = f"{ci.source_lean}/{ci.source_lean_basis}" if ci.source_lean else "?"
            print(f"  [{ci.opinion_type:25}] {ci.claim.source} ({ci.speaker_type}, {lean}): {ci.claim.text[:90]}")
 
 
if __name__ == "__main__":
    main()
 