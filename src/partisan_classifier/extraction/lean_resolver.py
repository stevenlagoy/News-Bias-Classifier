"""Find the political lean of subjects that SUBJECTS_ALIGNMENTS does not know."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from typing import Callable


LEAN_SCORE = {"far left": -1.0, "left": -1.0, "moderate left": -0.5, "center": 0.0, "moderate right": 0.5, "right": 1.0, "far right": 1.0}
NOMINATE_SCALE = 0.6 # dim1 of -0.6 / +0.6 maps to -1 / +1


def lean_to_score(label: str | None) -> float | None:
    return LEAN_SCORE.get(label.casefold().strip()) if label else None


def score_to_lean(score: float) -> str:
    if score <= -0.75:
        return "left"
    if score <= -0.25:
        return "moderate left"
    if score < 0.25:
        return "center"
    if score <= 0.75:
        return "moderate right"
    return "right"


def _clip(x: float) -> float:
    return max(-1.0, min(1.0, x))


@dataclass
class LeanRecord:
    key: str
    lean: str | None
    score: float | None = None # -1 .. +1
    confidence: float = 0.0
    method: str = "unknown" # table | article-cue | voteview | wikidata | llm | unknown
    evidence: str | None = None
    retrieved_at: str = ""
    extra: dict = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[’']s$", "", name.strip())).casefold()


# Article cues

LEFT_CUES = {"democratic", "democrat", "progressive", "liberal"}
RIGHT_CUES = {"republican", "gop", "conservative"}
CUE_SCORE = 0.8


def cue_lean(key: str, context: str) -> LeanRecord | None:
    words = {w.lower().rstrip("s") for w in re.findall(r"[A-Za-z]+", context or "")}
    words = {w[:-1] if w.endswith("s") and w[:-1] in LEFT_CUES | RIGHT_CUES else w for w in words}
    left, right = bool(words & LEFT_CUES), bool(words & RIGHT_CUES)
    if left == right:
        return None
    score = -CUE_SCORE if left else CUE_SCORE
    return LeanRecord(key, score_to_lean(score), score, 0.7, "article-cue", context.strip(), _now())


# VoteView (HSall_members.csv)
_TITLE_WORDS = {
    "sen", "senator", "rep", "representative", "president", "vice", "vice-president", "gov", "governor",
    "mayor", "dr", "secretary", "speaker", "ambassador", "congressman", "congresswoman", "majority",
    "minority", "leader", "mr", "ms", "mrs"
}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
_NICK = {
    "chuck": ("charles",), "bill": ("william",), "bob": ("robert",), "bobby": ("robert",), "mike": ("michael",),
    "jim": ("james",), "tom": ("thomas",), "dick": ("richard",), "joe": ("joseph",), "ted": ("edward", "theodore"),
    "liz": ("elizabeth",), "dan": ("daniel",), "dave": ("david",), "steve": ("steven", "stephen"),
    "tim": ("timothy",), "greg": ("gregory",), "chris": ("christopher",), "maggie": ("margaret",),
    "peggy": ("margaret",), "kathy": ("katherine", "kathleen"), "jeff": ("jefferson", "jeffrey"),
    "matt": ("matthew",), "nick": ("nicholas",), "andy": ("andrew",), "tony": ("anthony",), "ben": ("benjamin",),
    "rick": ("richard", "frederick"), "don": ("donald",), "ron": ("ronald",), "ken": ("kenneth",)
}
_LEADING_TITLES = re.compile(
    r"^(?:(?:the|former|acting|then|u\.s\.)\s+)*(?:(?:attorney general|press secretary|secretary of state|secretary|"
    r"vice[- ]president|president|prime minister|senator|sen\.?|representative|rep\.?|congressman|congresswoman|"
    r"governor|gov\.?|mayor|speaker|majority leader|minority leader|chairman|chairwoman|chair|judge|justice|general|"
    r"gen\.?|admiral|ambassador|chancellor|sheriff|director|commissioner|dr\.?|mr\.?|mrs\.?|ms\.?|miss)\s+)+", re.I)
PARTY_CODES = {"100": "Democratic", "200": "Republican"} # VoteView HSall_parties.csv
_STOP_NAMES = {"it", "they", "he", "she", "we", "i", "you", "this", "that", "there", "who", "what", "them", "his", "her"}
_PARTICLES = {"van", "von", "de", "del", "da", "di", "la", "le", "bin", "al", "st.", "mc", "mac"}
_ORG_WORDS = {
    "department", "news", "times", "post", "foundation", "institute", "party", "committee", "council",
    "corporation", "company", "association", "network", "university", "court", "police", "agency",
    "administration", "house", "senate", "congress", "bureau", "campaign", "fund", "union", "coalition",
    "caucus", "alliance", "movement", "pac", "inc", "corp", "group", "center", "centre", "league"
}
_POLITICAL_ORG_WORDS = {
    "party", "committee", "foundation", "institute", "news", "times", "post", "network", "pac",
    "union", "coalition", "caucus", "association", "alliance", "fund", "campaign", "movement",
    "league", "center", "centre"
}


def _person_like(name: str) -> bool:
    toks = name.replace("’s", "").split()
    if not 2 <= len(toks) <= 4 or any(any(ch.isdigit() for ch in t) for t in toks):
        return False
    if any(t.casefold().strip(".") in ORG_WORDS for t in toks):
        return False
    return all(t[:1].isupper() or t.casefold() in _PARTICLES for t in toks)


def _worth_org_lookup(name: str) -> bool:
    return len(name.split()) >= 2 and any(t.casefold().strip(".") in _POLITICAL_ORG_WORDS for t in name.split())


def clean_person_name(name: str) -> str:
    return _LEADING_TITLES.sub("", name.strip()).strip() or name.strip()


def _name_tokens(name: str) -> list[str]:
    toks = re.sub(r"[’']s\b", "", name).replace(",", " ").split()
    out = []
    for t in toks:
        base = t.strip(".").casefold()
        if base in _TITLE_WORDS or base in _SUFFIXES:
            continue
        out.append(t.strip("."))
    return out


def _variants(tok: str) -> set[str]:
    t = tok.casefold().strip(".")
    return {t, *_NICK.get(t, ())}


def _float(x) -> float | None:
    try: return float(x)
    except (TypeError, ValueError): return None


@dataclass
class _Member:
    icpsr: str
    bioname: str
    surname: str
    given: list[str]
    nick: str | None
    rows: list[tuple[int, str, str, float | None, float | None]] = field(default_factory=list)


def _flat(x) -> list[str]:
    if x is None: return []
    if isinstance(x, str):
        t = x.casefold().strip(".")
        return [t] if t else []
    return [t for item in x for t in _flat(item)]


def _given_ok(query: list[str], m: _Member) -> bool:
    bio = set(_flat(m.given) + _flat(m.nick))
    initials = "".join(t[0] for t in _flat(m.given))
    for q in query:
        if _variants(q) & bio: continue
        letters = re.sub(r"[^A-Za-z]", "", q).casefold()
        # Names as abbreviations, like JD Vance
        if letters and len(letters) <= 3 and q.replace(".", "").isupper() and initials.startswith(letters): continue
        # Abbreviated letter, like J. Edgar Hoover
        if len(letters) == 1 and any(g.startswith(letters) for g in bio): continue # type: ignore
        return False
    return True


class Voteview:
    """Members of Congress (and presidents) with party and DW-NOMINATE scores."""
 
    def __init__(self, csv_path: str | Path):
        members: dict[str, _Member] = {}
        with open(csv_path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                bio = (r.get("bioname") or "").strip()
                if "," not in bio or not r.get("icpsr"):
                    continue
                m = members.get(r["icpsr"])
                if m is None:
                    surname, rest = bio.split(",", 1)
                    nick = re.search(r"\(([^)]+)\)", rest)
                    given = [g.strip(".").casefold() for g in re.sub(r"\([^)]*\)", "", rest).split()]
                    m = members[r["icpsr"]] = _Member(r["icpsr"], bio, surname.strip().casefold(), given,
                                                     nick.group(1).casefold() if nick else None)
                try:
                    cong = int(r["congress"])
                except (KeyError, ValueError):
                    continue
                m.rows.append((cong, r.get("chamber", ""), str(r.get("party_code", "")),
                               _float(r.get("nominate_dim1")), _float(r.get("nominate_dim2"))))
        self.by_surname: dict[str, list[_Member]] = defaultdict(list)
        for m in members.values():
            m.rows.sort()
            self.by_surname[m.surname].append(m)
        self.max_congress = max((m.rows[-1][0] for m in members.values() if m.rows), default=0)
 
    @staticmethod
    def congress_of(year: int) -> int:
        return max(1, (year - 1789) // 2 + 1)
 
    def _row(self, m: _Member, congress: int | None):
        if congress is None:
            return m.rows[-1]
        before = [r for r in m.rows if r[0] <= congress]
        return before[-1] if before else m.rows[0]
 
    def _active(self, m: _Member, congress: int | None) -> bool:
        ref = congress or self.max_congress
        return any(ref - 1 <= r[0] <= ref for r in m.rows)
 
    def find(self, name: str, as_of_year: int | None = None):
        toks = _name_tokens(name)
        congress = self.congress_of(as_of_year) if as_of_year else None
        for n_sur in (1, 2): # "Van Hollen" is a two-word surname
            if len(toks) < n_sur:
                break
            cands = self.by_surname.get(" ".join(toks[-n_sur:]).casefold())
            given = toks[:-n_sur]
            if not cands:
                continue
            if given:
                ok = [m for m in cands if _given_ok(given, m)]
                if ok:
                    best = max(ok, key=lambda m: (congress is None or m.rows[0][0] <= congress,
                                                  self._row(m, congress)[0]))
                    return best, self._row(best, congress), 0.9
            else:
                active = [m for m in cands if self._active(m, congress)]
                if len(active) == 1: # a bare surname is only safe when it is unambiguous
                    return active[0], self._row(active[0], congress), 0.65
        return None
 
    def record(self, key: str, as_of_year: int | None = None) -> LeanRecord | None:
        hit = self.find(key, as_of_year)
        if hit is None:
            return None
        m, (cong, chamber, party_code, d1, d2), conf = hit
        party = PARTY_CODES.get(party_code, "other")
        extra = {"party": party, "congress": cong, "chamber": chamber, "nominate_dim1": d1, "nominate_dim2": d2}
        evidence = f"Voteview: {m.bioname}, Congress {cong} ({chamber})"
        if d1 is not None:
            score = _clip(d1 / NOMINATE_SCALE)
            return LeanRecord(key, score_to_lean(score), round(score, 3), conf, "voteview", evidence, _now(), extra)
        if party in {"Democratic", "Republican"}: # no score estimated: fall back to party, with less confidence
            score = -CUE_SCORE if party == "Democratic" else CUE_SCORE
            return LeanRecord(key, score_to_lean(score), score, conf * 0.6, "voteview", evidence, _now(), extra)
        return None

# Wikidata

PARTY_SCORE = {"democratic party": -0.8, "republican party": 0.8, "democratic-farmer-labor party": -0.8}
ALIGNMENT_SCORE = {
    "far-left": -1.0, "left-wing": -0.8, "centre-left": -0.4, "center-left": -0.4,
    "centrism": 0.0, "centre": 0.0, "center": 0.0,
    "centre-right": 0.4, "center-right": 0.4, "right-wing": 0.8, "far-right": 1.0
}
IDEOLOGY_SCORE = {
    "conservatism": 0.7, "american conservatism": 0.7, "social conservatism": 0.7,
    "fiscal conservatism": 0.6, "neoconservatism": 0.7, "right-wing populism": 0.7, "trumpism": 0.8,
    "progressivism": -0.7, "modern liberalism in the united states": -0.7, "social liberalism": -0.6,
    "social democracy": -0.7, "democratic socialism": -0.8, "socialism": -0.8
}
PERSON_WORDS = {
    "politician", "senator", "representative", "governor", "mayor", "president", "congressman",
    "congresswoman", "activist", "commentator", "businessman", "businesswoman", "lawyer", "journalist",
    "judge", "diplomat", "official", "statesman", "candidate", "strategist"
}
ORG_WORDS = {
    "party", "political", "organization", "organisation", "think", "tank", "company", "newspaper", "news",
    "network", "union", "committee", "movement", "coalition", "agency", "pac", "group", "foundation",
    "institute", "association", "outlet", "website"
}

class Wikidata:
    API = "https://www.wikidata.org/w/api.php"
    USER_AGENT = "News-Bias-Classifier/0.1 (course project; contact: you@example.edu)"
 
    def __init__(self, fetch: Callable[[str], dict] | None = None, pause: float = 0.2):
        self._fetch = fetch or self._http_json
        self._pause = pause
 
    def _http_json(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    time.sleep(self._pause)
                    return json.load(resp)
            except Exception:
                if attempt:
                    raise
                time.sleep(1.0)
        return {}
 
    def _api(self, **params) -> dict:
        return self._fetch(f"{self.API}?{urllib.parse.urlencode({**params, 'format': 'json'})}")
 
    def _labels(self, qids: list[str]) -> dict[str, str]:
        if not qids:
            return {}
        data = self._api(action="wbgetentities", ids="|".join(qids), props="labels", languages="en")
        return {q: e.get("labels", {}).get("en", {}).get("value", "") for q, e in data.get("entities", {}).items()}
 
    @staticmethod
    def _targets(claims: dict, prop: str) -> list[str]:
        """Entity IDs for a property, preferring statements without an end date (current ones)."""
        current, ended = [], []
        for st in claims.get(prop, []):
            snak = st.get("mainsnak", {})
            if snak.get("snaktype") != "value":
                continue
            qid = snak["datavalue"]["value"].get("id")
            (ended if "P582" in st.get("qualifiers", {}) else current).append(qid)
        return [q for q in (current or ended) if q]
 
    def lookup(self, name: str, is_person: bool, hint: str = "") -> dict | None:
        found = self._api(action="wbsearchentities", search=name, language="en", type="item", limit=7)
        hint_words = {w.casefold() for w in re.findall(r"[A-Za-z]{4,}", hint)}
        best, best_score = None, 0
        for cand in found.get("search", []):
            desc = (cand.get("description") or "").casefold()
            words = set(re.findall(r"[a-z]+", desc))
            score = len(words & (PERSON_WORDS if is_person else ORG_WORDS)) + len(words & hint_words)
            if score > best_score:
                best, best_score = cand, score
        if best is None:
            return None
        qid = best["id"]
        ent = self._api(action="wbgetentities", ids=qid, props="claims|descriptions|sitelinks",
                        sitefilter="enwiki", languages="en").get("entities", {}).get(qid, {})
        claims = ent.get("claims", {})
        human = "Q5" in self._targets(claims, "P31")
        if human != is_person:
            return None
        ids = {p: self._targets(claims, p) for p in ("P102", "P1387", "P1142")}
        labels = self._labels(sorted({q for v in ids.values() for q in v}))
        return {
            "qid": qid, "label": best.get("label", name), "description": best.get("description", ""),
            "party": [labels.get(q, "") for q in ids["P102"]],
            "alignment": [labels.get(q, "") for q in ids["P1387"]],
            "ideology": [labels.get(q, "") for q in ids["P1142"]],
            "wikipedia": ent.get("sitelinks", {}).get("enwiki", {}).get("title"),
        }
 
    def summary(self, title: str) -> str:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_"))
        return self._fetch(url).get("extract", "")
 
 
def _match(label: str, table: dict[str, float]) -> float | None:
    """Score for the longest table phrase contained in a Wikidata label ("right-wing politics" -> right-wing)."""
    l = label.casefold()
    hits = [k for k in table if k in l]
    return table[max(hits, key=len)] if hits else None
 
 
def wikidata_record(key: str, info: dict) -> LeanRecord | None:
    evidence = f"https://www.wikidata.org/wiki/{info['qid']}"
    extra = {k: info[k] for k in ("label", "description", "party", "alignment", "ideology")}
    for table, field_name, conf in ((PARTY_SCORE, "party", 0.75), (ALIGNMENT_SCORE, "alignment", 0.6),
                                    (IDEOLOGY_SCORE, "ideology", 0.55)):
        scores = [sc for l in info[field_name] if (sc := _match(l, table)) is not None]
        if scores:
            s = sum(scores) / len(scores)
            return LeanRecord(key, score_to_lean(s), round(s, 3), conf, "wikidata", evidence, _now(), extra)
    return None


# LLM over a retrieved passage

ALLOWED = ["left", "moderate left", "center", "moderate right", "right", "unknown"]
 
 
def llm_record(key: str, passage: str, generate: Callable[[str], str], source: str | None = None) -> LeanRecord | None:
    """`generate` maps a prompt to text, so any local model or API works. The answer is accepted only
    if it picks a known label AND quotes a span that really appears in the passage."""
    prompt = (
        f"Text:\n{passage}\n\nUsing only the text above, what is the political leaning of {key}? "
        f"Reply with JSON only: {{\"leaning\": one of {ALLOWED}, \"evidence\": \"an exact quote from the text\"}}"
    )
    m = re.search(r"\{.*\}", generate(prompt), re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    lean, quote = str(out.get("leaning", "")).casefold(), str(out.get("evidence", "")).strip()
    if lean not in LEAN_SCORE or len(quote) < 8 or quote.casefold() not in passage.casefold():
        return None
    return LeanRecord(key, lean, LEAN_SCORE[lean], 0.5, "llm", quote, _now(), {"source": source})



# Resolver

class LeanResolver:
    def __init__(
        self, voteview: Voteview | None = None, wikidata: Wikidata | None = None,
        cache_path: str | Path | None = None, llm: Callable[[str], str] | None = None,
        negative_ttl_days: int = 30
    ):
        self.voteview, self.wikidata, self.llm = voteview, wikidata, llm
        self.cache_path = Path(cache_path) if cache_path else None
        self.ttl = timedelta(days=negative_ttl_days)
        self._cache: dict[str, LeanRecord] = {}
        self._failures = 0
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                    ck = d.pop("_ck") # pop first: Python evaluates the right-hand side before the subscript
                    self._cache[ck] = LeanRecord(**d)
                except (ValueError, TypeError):
                    continue
 
    @staticmethod
    def _ck(name: str, is_person: bool | None, as_of_year: int | None) -> str:
        kind = "x" if is_person is None else "p" if is_person else "o"
        return f"{_norm(name)}|{kind}|{as_of_year // 2 if as_of_year else 'latest'}"

    def _remember(self, ck: str, rec: LeanRecord) -> LeanRecord:
        self._cache[ck] = rec
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"_ck": ck, **asdict(rec)}) + "\n")
        return rec
 
    def _fresh(self, rec: LeanRecord) -> bool:
        if rec.method != "unknown":
            return True
        try:
            return datetime.now(timezone.utc) - datetime.fromisoformat(rec.retrieved_at) < self.ttl
        except ValueError:
            return False
 
    def resolve(self, name: str, *, is_person: bool | None = True, context: str = "", as_of_year: int | None = None, online: bool = True) -> LeanRecord:
        if name.strip().casefold() in _STOP_NAMES or len(name.strip()) <= 2:
            return LeanRecord(name, None, None, 0.0, "unknown", "not a name", _now())
        cue = cue_lean(name, context)
        if cue:
            return cue # article-specific, so never cached
        person_name = clean_person_name(name)
        if is_person is None:
            is_person = _person_like(person_name)
        ck = self._ck(name, is_person, as_of_year)
        hit = self._cache.get(ck)
        if hit and self._fresh(hit):
            return hit
        if is_person and self.voteview:
            try:
                rec = self.voteview.record(person_name, as_of_year)
            except Exception:
                rec = None 
            if rec:
                rec.key = name
                return self._remember(ck, rec)
        asked_network = False
        target = person_name if is_person else name
        if len(target.split()) >= 2 and (is_person or _worth_org_lookup(name)):
            wk = f"wd|{_norm(target)}|{'p' if is_person else 'o'}"   # Wikidata does not depend on the article's year
            wh = self._cache.get(wk)
            if wh and self._fresh(wh):          # a cache hit works even with no network configured
                if wh.lean:
                    return wh
            elif online and self.wikidata:
                asked_network = True
                try:
                    info = self.wikidata.lookup(target, is_person, context)
                    self._failures = 0
                except Exception:               # network trouble: answer unknown now, retry later
                    self._failures += 1
                    if self._failures >= 6 and self.wikidata is not None:
                        print("Wikidata is not answering; continuing without it for this run "
                              "(cached results are kept).", file=sys.stderr)
                        self.wikidata = None
                    return LeanRecord(name, None, None, 0.0, "unknown", "lookup failed", _now())
                rec = wikidata_record(name, info) if info else None
                if rec is None and info and self.llm and info.get("wikipedia"):
                    try:
                        rec = llm_record(name, self.wikidata.summary(info["wikipedia"]), self.llm, info["wikipedia"])
                    except Exception:
                        rec = None
                self._remember(wk, rec or LeanRecord(name, None, None, 0.0, "unknown", None, _now()))
                if rec:
                    return self._remember(ck, rec)
        unknown = LeanRecord(name, None, None, 0.0, "unknown", None, _now())
        if asked_network or (online and self.wikidata):
            return self._remember(ck, unknown)
        return unknown

 
def build_resolver(
    voteview_csv: str | None = None, online: bool = False, cache_path: str | None = None,
    llm: Callable[[str], str] | None = None
) -> LeanResolver:
    return LeanResolver(
        Voteview(voteview_csv) if voteview_csv else None,
        Wikidata() if online else None,
        cache_path, llm
    )
