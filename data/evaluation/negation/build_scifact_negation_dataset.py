import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from hashlib import md5
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = '/mnt/data'
CORPUS_IN = os.path.join(DATA_DIR, 'corpus.jsonl')
QUERIES_IN = os.path.join(DATA_DIR, 'queries.jsonl')
QRELS_IN = [os.path.join(DATA_DIR, 'qrels_train.jsonl'), os.path.join(DATA_DIR, 'qrels_test.jsonl')]
CORPUS_OUT = os.path.join(DATA_DIR, 'scifact_negation_corpus.jsonl')
META_OUT = os.path.join(DATA_DIR, 'scifact_negation_metadata.jsonl')
AUDIT_OUT = os.path.join(DATA_DIR, 'scifact_negation_audit.jsonl')
STATS_OUT = os.path.join(DATA_DIR, 'scifact_negation_stats.json')

random.seed(42)
np.random.seed(42)

GENERIC_STOP = set(
    'study studies evidence effect effects patient patients people population related relation relatedly results result data model models analysis review reviews trial trials treatment treatments disease diseases rate rates level levels outcome outcomes'.split()
)
STOP = set(ENGLISH_STOP_WORDS) | GENERIC_STOP

POLARITY_WORDS = set(
    'not no without lack lacks lacking fail fails failed absence absent neither never non ineffective unrelated independent noassociation noeffect '
    'does doesnt didnt cannot cant unable prevents prevent prevented prevention reduce reduces reduced reduction reductions lower lowers lowered low '
    'decreases decreased decrease smaller less lesser suppress suppresses suppressed inhibition inhibits inhibited inhibit negative negatively worse worsens worsening '
    'harm harmful impaired attenuation attenuate attenuated attenuates loss losses insufficient insufficiency neutral similar same unchanged stable '
    'increase increases increased increasing higher high more greater elevated elevate gain gains gained improves improved improvement beneficial benefit benefits '
    'effective efficacy sensitive sensitivity specific susceptible promotes promoted promote induces induced induce activates activated activate required necessary essential important crucial '
    'positive positively better association associated risk risks'.split()
)

NEGATION_MARKERS = set('not no without lack lacks lacking fail fails failed absence absent independent unrelated none neither never'.split())

CUE_MAP = {
    'up': {'increase','increases','increased','higher','high','greater','elevated','elevate','gain','gains','gained','promote','promotes','promoted','induce','induces','induced','activate','activates','activated','raise','raises','raised','enhance','enhances','enhanced','susceptible'},
    'down': {'decrease','decreases','decreased','lower','low','less','lesser','reduce','reduces','reduced','reduction','reductions','prevent','prevents','prevented','protect','protects','protected','suppress','suppresses','suppressed','inhibit','inhibits','inhibited','attenuate','attenuates','attenuated','improve','improves','improved','beneficial','benefit','better','effective','efficacy','sensitivity','sensitive'},
    'null': {'not','no','without','lack','lacks','lacking','fails','failed','fail','absence','absent','independent','unrelated','unchanged','similar','same','none'},
    'need': {'required','necessary','essential','important','crucial'},
    'noneed': {'dispensable','unnecessary','independent','notrequired'},
    'posval': {'positive','beneficial','better','improved','effective','protective'},
    'negval': {'negative','harmful','worse','adverse','impaired','bad'},
    'more': {'more','higher','greater','larger'},
    'less': {'less','lower','smaller'},
}

OPP_FAMILY = {
    'up': ['down','null'],
    'down': ['up','null'],
    'null': ['up','down'],
    'need': ['noneed','null'],
    'noneed': ['need'],
    'posval': ['negval','null'],
    'negval': ['posval','null'],
    'more': ['less','null'],
    'less': ['more','null'],
}

FAMILY_PRIORITY = ['down','need','posval','more','up','null','noneed','negval','less']


def norm_text(s: str) -> str:
    s = s.lower().replace('-', ' ')
    s = s.replace('β', ' beta ').replace('α', ' alpha ').replace('κ', ' kappa ')
    return re.sub(r'[^a-z0-9\s]', ' ', s)


def tokenize(s: str) -> List[str]:
    return [t for t in norm_text(s).split() if t]


def content_tokens(s: str) -> List[str]:
    return [t for t in tokenize(s) if t not in STOP]


def core_tokens(s: str) -> List[str]:
    return [t for t in content_tokens(s) if t not in POLARITY_WORDS]


def has_negation(s: str) -> bool:
    toks = set(tokenize(s))
    return bool(toks & NEGATION_MARKERS)


def detect_families(s: str) -> List[str]:
    toks = set(tokenize(s))
    out = []
    for fam, cues in CUE_MAP.items():
        if toks & cues:
            out.append(fam)
    if has_negation(s) and 'null' not in out:
        out.append('null')
    return out


def dominant_family(s: str) -> Optional[str]:
    fams = detect_families(s)
    for fam in FAMILY_PRIORITY:
        if fam in fams:
            return fam
    return None


def lower_first(s: str) -> str:
    if not s:
        return s
    return s[0].lower() + s[1:] if s[0].isalpha() else s


def strip_period(s: str) -> str:
    return s.strip().rstrip('.')


def split_sentences(text: str) -> List[str]:
    text = re.sub(
        r'\b(BACKGROUND|METHODS|RESULTS|CONCLUSIONS?|OBJECTIVE|OBJECTIVES|FINDINGS|INTERPRETATION|DESIGN|DATA SOURCES|STUDY SELECTION|DATA SYNTHESIS|MAIN OUTCOME MEASURE|PURPOSE|AIMS|AIM|CONTEXT)\b',
        r'\1. ',
        text,
    )
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
    out = []
    for p in parts:
        p = p.strip()
        if not p or len(p) < 2:
            continue
        out.append(p)
    if not out:
        out = [text.strip()] if text.strip() else []
    return out


def heavy_numeric(q: str) -> bool:
    low = q.lower()
    return bool(re.search(r'\b\d', low)) or '%' in low or 'mg' in low or 'hour' in low or 'year' in low


def safe_to_negate(q: str) -> bool:
    low = q.lower()
    if heavy_numeric(q):
        return False
    bad = ['more than', 'less than', 'at least', 'at most', 'compared with', 'versus', 'vs.', ' vs ']
    if any(b in low for b in bad):
        return False
    return True


def to_negation(claim: str) -> str:
    s = strip_period(claim)
    low = s.lower()
    if has_negation(low):
        return s
    rules = [
        (r'\b is associated with \b', ' is not associated with '),
        (r'\b are associated with \b', ' are not associated with '),
        (r'\b was associated with \b', ' was not associated with '),
        (r'\b were associated with \b', ' were not associated with '),
        (r'\b has no association with \b', ' is associated with '),
        (r'\b have no association with \b', ' are associated with '),
        (r'\b leads to \b', ' does not lead to '),
        (r'\b lead to \b', ' do not lead to '),
        (r'\b causes \b', ' does not cause '),
        (r'\b cause \b', ' do not cause '),
        (r'\b increases \b', ' does not increase '),
        (r'\b increase \b', ' do not increase '),
        (r'\b decreases \b', ' does not decrease '),
        (r'\b decrease \b', ' do not decrease '),
        (r'\b reduces \b', ' does not reduce '),
        (r'\b reduce \b', ' do not reduce '),
        (r'\b improves \b', ' does not improve '),
        (r'\b improve \b', ' do not improve '),
        (r'\b prevents \b', ' does not prevent '),
        (r'\b prevent \b', ' do not prevent '),
        (r'\b enhances \b', ' does not enhance '),
        (r'\b enhance \b', ' do not enhance '),
        (r'\b suppresses \b', ' does not suppress '),
        (r'\b suppress \b', ' do not suppress '),
        (r'\b activates \b', ' does not activate '),
        (r'\b activate \b', ' do not activate '),
        (r'\b promotes \b', ' does not promote '),
        (r'\b promote \b', ' do not promote '),
        (r'\b induces \b', ' does not induce '),
        (r'\b induce \b', ' do not induce '),
        (r'\b is required for \b', ' is not required for '),
        (r'\b are required for \b', ' are not required for '),
        (r'\b is necessary for \b', ' is not necessary for '),
        (r'\b are necessary for \b', ' are not necessary for '),
        (r'\b is important for \b', ' is not required for '),
        (r'\b are important for \b', ' are not required for '),
        (r'\b predicts \b', ' does not predict '),
        (r'\b predict \b', ' do not predict '),
        (r'\b predictive of \b', ' not predictive of '),
        (r'\b correlates with \b', ' does not correlate with '),
        (r'\b correlate with \b', ' do not correlate with '),
        (r'\b can be \b', ' cannot be '),
        (r'\b can \b', ' cannot '),
        (r'\b shows \b', ' does not show '),
        (r'\b show \b', ' do not show '),
    ]
    for pat, rep in rules:
        if re.search(pat, low):
            return re.sub(pat, rep, s, count=1, flags=re.I)
    # fallback on auxiliary insertion
    if re.search(r'\b is \b', low):
        return re.sub(r'\b is \b', ' is not ', s, count=1, flags=re.I)
    if re.search(r'\b are \b', low):
        return re.sub(r'\b are \b', ' are not ', s, count=1, flags=re.I)
    if re.search(r'\b has \b', low):
        return re.sub(r'\b has \b', ' does not have ', s, count=1, flags=re.I)
    if re.search(r'\b have \b', low):
        return re.sub(r'\b have \b', ' do not have ', s, count=1, flags=re.I)
    return 'No evidence that ' + lower_first(s)


def build_variants(q_pos: str, q_neg: str) -> List[Tuple[int, str, str]]:
    q_pos = strip_period(q_pos)
    q_neg = strip_period(q_neg)
    v1 = q_neg
    v2 = 'evidence that ' + lower_first(q_neg)
    if q_neg.lower().startswith('no evidence that'):
        v3 = 'lack of evidence that ' + lower_first(q_pos)
    else:
        v3 = 'no evidence that ' + lower_first(q_pos)
    variants = [
        (1, 'direct_negation', v1),
        (2, 'literature_need', v2),
        (3, 'paraphrastic_negation', v3),
    ]
    # ensure uniqueness in edge cases
    seen = set()
    unique = []
    for vid, vtype, text in variants:
        txt = re.sub(r'\s+', ' ', text).strip()
        if txt.lower() in seen:
            if vtype == 'literature_need':
                txt = 'studies showing that ' + lower_first(q_neg)
            elif vtype == 'paraphrastic_negation':
                txt = 'studies reporting no evidence that ' + lower_first(q_pos)
        seen.add(txt.lower())
        unique.append((vid, vtype, txt))
    return unique


@dataclass
class Unit:
    kind: str  # pair or single
    core_key: Tuple[str, ...]
    source_qids: List[str]
    q_pos: str
    q_sat: str  # claim whose support implies satisfaction of negation query
    q_neg: str
    core_terms: List[str]
    source_meta: Dict


def choose_pair_orientation(q1: str, q2: str) -> Tuple[str, str, str]:
    # returns q_pos, q_sat, q_neg
    n1, n2 = has_negation(q1), has_negation(q2)
    if n1 and not n2:
        return q2, q1, strip_period(q1)
    if n2 and not n1:
        return q1, q2, strip_period(q2)

    f1, f2 = set(detect_families(q1)), set(detect_families(q2))

    def score_as_positive(fams: set, text: str) -> int:
        score = 0
        if 'down' in fams: score += 4
        if 'need' in fams: score += 3
        if 'posval' in fams: score += 3
        if 'more' in fams: score += 2
        if 'up' in fams: score += 1
        if 'null' in fams: score -= 4
        if 'noneed' in fams: score -= 3
        if 'negval' in fams: score -= 3
        if 'less' in fams: score -= 1
        if heavy_numeric(text): score -= 2
        return score

    s1 = score_as_positive(f1, q1)
    s2 = score_as_positive(f2, q2)
    if s2 > s1:
        q_pos, q_sat = q2, q1
    elif s1 > s2:
        q_pos, q_sat = q1, q2
    else:
        # shorter query as affirmative anchor, deterministic
        q_pos, q_sat = (q1, q2) if len(q1) <= len(q2) else (q2, q1)
    q_neg = to_negation(q_pos)
    return q_pos, q_sat, q_neg


def pair_specific_tokens(a: str, b: str) -> Tuple[List[str], List[str]]:
    ta = [t for t in content_tokens(a) if t not in STOP]
    tb = [t for t in content_tokens(b) if t not in STOP]
    sa, sb = set(ta), set(tb)
    da = [t for t in ta if t in (sa - sb)]
    db = [t for t in tb if t in (sb - sa)]
    return da[:6], db[:6]


# ---------- Load raw data ----------
print('Loading queries/corpus/qrels...')
queries: Dict[str, str] = {}
for line in open(QUERIES_IN):
    obj = json.loads(line)
    queries[obj['_id']] = strip_period(obj['text'])

qrels: Dict[str, set] = defaultdict(set)
for fn in QRELS_IN:
    for line in open(fn):
        obj = json.loads(line)
        qrels[obj['query-id']].add(obj['corpus-id'])

corpus: Dict[str, Dict[str, str]] = {}
ordered_doc_ids: List[str] = []
for line in open(CORPUS_IN):
    obj = json.loads(line)
    corpus[obj['_id']] = {'title': obj['title'], 'text': obj['text']}
    ordered_doc_ids.append(obj['_id'])

# ---------- Sentence index ----------
print('Building sentence index...')
doc_sentences: Dict[str, List[str]] = {}
sent_texts: List[str] = []
sent_doc_ids: List[str] = []
sent_local_idx: List[int] = []
doc_to_sent_indices: Dict[str, List[int]] = defaultdict(list)
for did in ordered_doc_ids:
    sents = split_sentences(corpus[did]['text'])
    doc_sentences[did] = sents
    for j, sent in enumerate(sents):
        idx = len(sent_texts)
        sent_texts.append(corpus[did]['title'] + ' ' + sent)
        sent_doc_ids.append(did)
        sent_local_idx.append(j)
        doc_to_sent_indices[did].append(idx)

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.9, min_df=1)
X = vectorizer.fit_transform(sent_texts)


@lru_cache(maxsize=None)
def claim_simvec(claim: str) -> np.ndarray:
    qv = vectorizer.transform([claim])
    sims = cosine_similarity(qv, X).ravel().astype(np.float32)
    return sims


def top_unique_docs_for_claim(claim: str, max_docs: int = 40, max_sents: int = 250) -> List[str]:
    sims = claim_simvec(claim)
    if max_sents >= len(sims):
        idxs = np.argsort(sims)[::-1]
    else:
        topk = np.argpartition(sims, -max_sents)[-max_sents:]
        idxs = topk[np.argsort(sims[topk])[::-1]]
    out = []
    seen = set()
    for idx in idxs:
        did = sent_doc_ids[int(idx)]
        if did in seen:
            continue
        seen.add(did)
        out.append(did)
        if len(out) >= max_docs:
            break
    return out


def best_sentence_for_doc(claim: str, did: str) -> Tuple[float, Optional[int]]:
    sims = claim_simvec(claim)
    best_idx = None
    best_score = -1.0
    for idx in doc_to_sent_indices[did]:
        score = float(sims[idx])
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_score if best_score >= 0 else 0.0, best_idx


def sentence_from_global_idx(idx: Optional[int]) -> str:
    if idx is None:
        return ''
    return doc_sentences[sent_doc_ids[idx]][sent_local_idx[idx]]


def doc_core_coverage(did: str, core: List[str]) -> int:
    text = norm_text(corpus[did]['title'] + ' ' + corpus[did]['text'])
    toks = set(text.split())
    return sum(1 for t in core if t in toks)


def doc_text_tokens(did: str) -> set:
    return set(norm_text(corpus[did]['title'] + ' ' + corpus[did]['text']).split())

# ---------- Candidate semantic units ----------
print('Selecting semantic units...')
all_qids = list(queries.keys())
by_core: Dict[Tuple[str, ...], List[Tuple[str, str, float]]] = defaultdict(list)
for i, qa_id in enumerate(all_qids):
    qa = queries[qa_id]
    ca = set(core_tokens(qa))
    if len(ca) < 2:
        continue
    for qb_id in all_qids[i+1:]:
        qb = queries[qb_id]
        cb = set(core_tokens(qb))
        inter = ca & cb
        if len(inter) < 2:
            continue
        jac = len(inter) / max(len(ca | cb), 1)
        if jac < 0.5:
            continue
        if not ((set(detect_families(qa)) != set(detect_families(qb))) or (has_negation(qa) != has_negation(qb))):
            continue
        core = tuple(sorted(inter))
        by_core[core].append((qa_id, qb_id, jac))

pair_units: List[Unit] = []
used_qids = set()
for core_key, vals in by_core.items():
    # prefer stronger lexical overlap and shorter, less numeric claims
    def pair_rank(v):
        a, b, jac = v
        q1, q2 = queries[a], queries[b]
        rank = jac
        rank += 0.1 if has_negation(q1) or has_negation(q2) else 0
        rank -= 0.2 if heavy_numeric(q1) or heavy_numeric(q2) else 0
        rank += 0.01 * len(core_key)
        return rank
    a, b, jac = max(vals, key=pair_rank)
    q1, q2 = queries[a], queries[b]
    if heavy_numeric(q1) or heavy_numeric(q2):
        continue
    if len(core_key) < 3:
        continue
    q_pos, q_sat, q_neg = choose_pair_orientation(q1, q2)
    unit = Unit(
        kind='pair',
        core_key=core_key,
        source_qids=[a, b],
        q_pos=q_pos,
        q_sat=q_sat,
        q_neg=q_neg,
        core_terms=list(core_key),
        source_meta={'pair_jaccard': jac, 'q1': q1, 'q2': q2},
    )
    pair_units.append(unit)
    used_qids.update([a, b])

# We expect ~193 pair units after the filters.
pair_units.sort(key=lambda u: (-len(u.core_terms), u.q_neg.lower()))
print(f'Pair units kept: {len(pair_units)}')

# Add lightweight single units to reach 200 semantic units and 600 total query variants.
# Keep this stage cheap: rank by lexical suitability and existing topical qrels, not by retrieval.
single_candidates: List[Tuple[float, Unit]] = []
for qid, q in queries.items():
    if qid in used_qids:
        continue
    if not safe_to_negate(q):
        continue
    core = sorted(set(core_tokens(q)))
    if len(core) < 3:
        continue
    q_pos = q
    q_neg = to_negation(q)
    if q_neg.lower() == q_pos.lower():
        continue
    fam = set(detect_families(q_pos))
    score = 0.0
    score += 0.05 * min(len(qrels.get(qid, set())), 3)
    score += 0.01 * len(core)
    if 'down' in fam or 'up' in fam:
        score += 0.08
    if 'need' in fam or 'posval' in fam:
        score += 0.06
    if has_negation(q_pos):
        score -= 0.1
    unit = Unit(
        kind='single',
        core_key=tuple(core),
        source_qids=[qid],
        q_pos=q_pos,
        q_sat=q_neg,
        q_neg=q_neg,
        core_terms=core,
        source_meta={'source_query': q},
    )
    single_candidates.append((score, unit))

single_candidates.sort(key=lambda x: (-x[0], -len(x[1].core_terms), x[1].q_neg.lower()))
selected_single_units = [u for _, u in single_candidates[: max(0, 200 - len(pair_units))]]
print(f'Single units added: {len(selected_single_units)}')

semantic_units: List[Unit] = pair_units + selected_single_units
semantic_units = semantic_units[:200]
print(f'Total semantic units: {len(semantic_units)}')

assert len(semantic_units) == 200, f'Expected 200 semantic units, got {len(semantic_units)}'

# ---------- Doc labeling ----------
print('Constructing query-level judgments...')


def select_irrelevant_docs(core_terms: List[str], excluded: set, how_many: int = 2) -> List[str]:
    core_set = set(core_terms)
    seed_int = int(md5(' '.join(sorted(core_terms)).encode('utf-8')).hexdigest()[:8], 16)
    start = seed_int % len(ordered_doc_ids)
    out = []
    for offset in range(len(ordered_doc_ids)):
        did = ordered_doc_ids[(start + offset) % len(ordered_doc_ids)]
        if did in excluded:
            continue
        if core_set & doc_text_tokens(did):
            continue
        out.append(did)
        if len(out) >= how_many:
            break
    # fallback if the corpus happens to match too much
    if len(out) < how_many:
        for did in ordered_doc_ids:
            if did in excluded or did in out:
                continue
            out.append(did)
            if len(out) >= how_many:
                break
    return out


def build_doc_judgments(unit: Unit) -> Dict:
    q_pos, q_sat, q_neg = unit.q_pos, unit.q_sat, unit.q_neg
    core = unit.core_terms
    source_qids = unit.source_qids
    qrel_pool = set()
    for qid in source_qids:
        qrel_pool |= set(qrels.get(qid, set()))

    candidate_docs = set(qrel_pool)
    for claim in [q_pos, q_sat, q_neg]:
        candidate_docs.update(top_unique_docs_for_claim(claim, max_docs=35))

    pos_diff_tokens, sat_diff_tokens = pair_specific_tokens(q_pos, q_sat)
    pos_fam = dominant_family(q_pos)
    sat_fam = dominant_family(q_sat) or dominant_family(q_neg)

    scored = []
    cov_thresh = 2 if len(core) >= 4 else 1
    for did in candidate_docs:
        pos_score, pos_idx = best_sentence_for_doc(q_pos, did)
        sat_score1, sat_idx1 = best_sentence_for_doc(q_sat, did)
        sat_score2, sat_idx2 = best_sentence_for_doc(q_neg, did)
        if sat_score2 > sat_score1:
            sat_score, sat_idx = sat_score2, sat_idx2
        else:
            sat_score, sat_idx = sat_score1, sat_idx1
        coverage = doc_core_coverage(did, core)
        toks = doc_text_tokens(did)
        pos_bonus = 0.02 * coverage
        sat_bonus = 0.02 * coverage
        pos_hits = [t for t in pos_diff_tokens if t in toks]
        sat_hits = [t for t in sat_diff_tokens if t in toks]
        pos_bonus += 0.01 * len(pos_hits)
        sat_bonus += 0.01 * len(sat_hits)
        if pos_fam and pos_fam in CUE_MAP and toks & CUE_MAP[pos_fam]:
            pos_bonus += 0.01
        if sat_fam and sat_fam in CUE_MAP and toks & CUE_MAP[sat_fam]:
            sat_bonus += 0.01
        if has_negation(q_neg) and toks & NEGATION_MARKERS:
            sat_bonus += 0.005

        pos_conf = pos_score + pos_bonus
        sat_conf = sat_score + sat_bonus
        scored.append({
            'did': did,
            'pos_score': float(pos_score),
            'sat_score': float(sat_score),
            'pos_conf': float(pos_conf),
            'sat_conf': float(sat_conf),
            'coverage': int(coverage),
            'pos_idx': pos_idx,
            'sat_idx': sat_idx,
            'pos_sentence': sentence_from_global_idx(pos_idx),
            'sat_sentence': sentence_from_global_idx(sat_idx),
            'title': corpus[did]['title'],
        })

    sat_candidates = []
    vio_candidates = []
    partial_candidates = []
    for row in sorted(scored, key=lambda r: max(r['pos_conf'], r['sat_conf']), reverse=True):
        if row['coverage'] < cov_thresh and row['did'] not in qrel_pool:
            continue
        diff = row['sat_conf'] - row['pos_conf']
        if row['sat_conf'] >= 0.11 and diff >= 0.015:
            sat_candidates.append(row)
        elif row['pos_conf'] >= 0.11 and -diff >= 0.015:
            vio_candidates.append(row)
        elif max(row['sat_conf'], row['pos_conf']) >= 0.10:
            partial_candidates.append(row)

    # Fallbacks so each semantic unit has both satisfying and violating docs.
    if not sat_candidates:
        sat_fallbacks = sorted(
            [r for r in scored if r['did'] not in {x['did'] for x in vio_candidates} and (r['coverage'] >= 1 or r['did'] in qrel_pool)],
            key=lambda r: (r['sat_conf'] - r['pos_conf'], r['sat_conf']),
            reverse=True,
        )
        if sat_fallbacks:
            sat_candidates.append(sat_fallbacks[0])
    if not vio_candidates:
        vio_fallbacks = sorted(
            [r for r in scored if r['did'] not in {x['did'] for x in sat_candidates} and (r['coverage'] >= 1 or r['did'] in qrel_pool)],
            key=lambda r: (r['pos_conf'] - r['sat_conf'], r['pos_conf']),
            reverse=True,
        )
        if vio_fallbacks:
            vio_candidates.append(vio_fallbacks[0])

    # choose up to 2 stronger docs on each side; with weak supervision, keep pools small
    sat_selected = []
    seen = set()
    for row in sorted(sat_candidates, key=lambda r: (r['sat_conf'] - r['pos_conf'], r['sat_conf']), reverse=True):
        if row['did'] in seen:
            continue
        sat_selected.append(row)
        seen.add(row['did'])
        if len(sat_selected) >= 2:
            break
    vio_selected = []
    for row in sorted(vio_candidates, key=lambda r: (r['pos_conf'] - r['sat_conf'], r['pos_conf']), reverse=True):
        if row['did'] in seen:
            continue
        vio_selected.append(row)
        seen.add(row['did'])
        if len(vio_selected) >= 2:
            break

    # If one side ended up empty after deduplication, backfill with best alternative.
    if not sat_selected:
        for row in sorted(scored, key=lambda r: (r['sat_conf'] - r['pos_conf'], r['sat_conf']), reverse=True):
            if row['did'] in seen:
                continue
            sat_selected.append(row)
            seen.add(row['did'])
            break
    if not vio_selected:
        for row in sorted(scored, key=lambda r: (r['pos_conf'] - r['sat_conf'], r['pos_conf']), reverse=True):
            if row['did'] in seen:
                continue
            vio_selected.append(row)
            seen.add(row['did'])
            break

    partial_selected = []
    for row in sorted(partial_candidates, key=lambda r: max(r['sat_conf'], r['pos_conf']), reverse=True):
        if row['did'] in seen:
            continue
        partial_selected.append(row)
        seen.add(row['did'])
        if len(partial_selected) >= 1:
            break

    topical = [r['did'] for r in sat_selected + vio_selected + partial_selected]
    irrelevant = select_irrelevant_docs(core, set(topical), how_many=2)

    graded = {}
    for row in sat_selected:
        graded[f'scifact-doc-{row["did"]}'] = 2
    for row in vio_selected:
        graded[f'scifact-doc-{row["did"]}'] = 0
    for row in partial_selected:
        graded[f'scifact-doc-{row["did"]}'] = 1
    for did in irrelevant:
        graded[f'scifact-doc-{did}'] = 0

    audit = {
        'source_kind': unit.kind,
        'source_qids': unit.source_qids,
        'q_pos': q_pos,
        'q_sat': q_sat,
        'q_neg': q_neg,
        'core_terms': core,
        'selected_satisfying': [
            {
                'doc_id': f'scifact-doc-{r["did"]}',
                'source_doc_id': int(r['did']),
                'title': r['title'],
                'sat_score': round(r['sat_score'], 4),
                'pos_score': round(r['pos_score'], 4),
                'sat_conf': round(r['sat_conf'], 4),
                'pos_conf': round(r['pos_conf'], 4),
                'coverage': r['coverage'],
                'evidence_sentence': r['sat_sentence'],
            }
            for r in sat_selected
        ],
        'selected_violating': [
            {
                'doc_id': f'scifact-doc-{r["did"]}',
                'source_doc_id': int(r['did']),
                'title': r['title'],
                'sat_score': round(r['sat_score'], 4),
                'pos_score': round(r['pos_score'], 4),
                'sat_conf': round(r['sat_conf'], 4),
                'pos_conf': round(r['pos_conf'], 4),
                'coverage': r['coverage'],
                'evidence_sentence': r['pos_sentence'],
            }
            for r in vio_selected
        ],
        'selected_partial': [
            {
                'doc_id': f'scifact-doc-{r["did"]}',
                'source_doc_id': int(r['did']),
                'title': r['title'],
                'sat_score': round(r['sat_score'], 4),
                'pos_score': round(r['pos_score'], 4),
                'sat_conf': round(r['sat_conf'], 4),
                'pos_conf': round(r['pos_conf'], 4),
                'coverage': r['coverage'],
                'sat_sentence': r['sat_sentence'],
                'pos_sentence': r['pos_sentence'],
            }
            for r in partial_selected
        ],
        'selected_irrelevant': [
            {
                'doc_id': f'scifact-doc-{did}',
                'source_doc_id': int(did),
                'title': corpus[did]['title'],
            }
            for did in irrelevant
        ],
    }

    return {
        'topical_relevant_doc_ids': [f'scifact-doc-{did}' for did in topical],
        'satisfying_doc_ids': [f'scifact-doc-{r["did"]}' for r in sat_selected],
        'violating_doc_ids': [f'scifact-doc-{r["did"]}' for r in vio_selected],
        'judged_irrelevant_doc_ids': [f'scifact-doc-{did}' for did in irrelevant],
        'graded_relevance': graded,
        'audit': audit,
    }


# ---------- Write corpus ----------
print('Writing corpus file...')
with open(CORPUS_OUT, 'w') as f:
    for did in ordered_doc_ids:
        obj = corpus[did]
        row = {
            'doc_id': f'scifact-doc-{did}',
            'source_dataset': 'scifact',
            'source_doc_id': int(did),
            'title': obj['title'],
            'text': obj['text'],
            'abstract_sentences': doc_sentences[did],
            'structured': False,
        }
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

# ---------- Write metadata + audit ----------
print('Writing metadata and audit files...')
meta_rows = 0
all_meta = []
all_audit = []
for idx, unit in enumerate(semantic_units, start=1):
    semantic_unit_id = f'scifact-neg-su{idx:05d}'
    judgments = build_doc_judgments(unit)
    variants = build_variants(unit.q_pos, unit.q_neg)
    audit_row = {
        'semantic_unit_id': semantic_unit_id,
        'kind': unit.kind,
        'source_qids': [int(q) for q in unit.source_qids],
        'q_pos': unit.q_pos,
        'q_sat': unit.q_sat,
        'q_neg': unit.q_neg,
        'core_terms': unit.core_terms,
        'source_meta': unit.source_meta,
        'audit': judgments['audit'],
    }
    all_audit.append(audit_row)
    for vid, vtype, qtext in variants:
        meta_rows += 1
        row = {
            'query_id': f'scifact-neg-q{meta_rows:06d}',
            'query': qtext,
            'source_dataset': 'scifact',
            'constraint_type': 'negation',
            'semantic_unit_id': semantic_unit_id,
            'query_variant_id': vid,
            'query_variant_type': vtype,
            'source_claim_ids': [int(q) for q in unit.source_qids],
            'topical_relevant_doc_ids': judgments['topical_relevant_doc_ids'],
            'satisfying_doc_ids': judgments['satisfying_doc_ids'],
            'violating_doc_ids': judgments['violating_doc_ids'],
            'judged_irrelevant_doc_ids': judgments['judged_irrelevant_doc_ids'],
            'graded_relevance': judgments['graded_relevance'],
        }
        all_meta.append(row)

with open(META_OUT, 'w') as f:
    for row in all_meta:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
with open(AUDIT_OUT, 'w') as f:
    for row in all_audit:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

# ---------- Validation ----------
print('Validating outputs...')
corpus_ids = {f'scifact-doc-{did}' for did in ordered_doc_ids}
query_ids = set()
semantic_counts = defaultdict(int)
for row in all_meta:
    assert row['query_id'] not in query_ids
    query_ids.add(row['query_id'])
    semantic_counts[row['semantic_unit_id']] += 1
    assert row['source_dataset'] == 'scifact'
    assert row['constraint_type'] == 'negation'
    assert row['satisfying_doc_ids'], row['query_id']
    assert row['violating_doc_ids'], row['query_id']
    assert row['judged_irrelevant_doc_ids'], row['query_id']
    s = set(row['satisfying_doc_ids'])
    v = set(row['violating_doc_ids'])
    irr = set(row['judged_irrelevant_doc_ids'])
    assert not (s & v), row['query_id']
    assert not (s & irr), row['query_id']
    assert not (v & irr), row['query_id']
    for did in s | v | irr | set(row['topical_relevant_doc_ids']):
        assert did in corpus_ids, (row['query_id'], did)
    for did, grade in row['graded_relevance'].items():
        assert did in corpus_ids
        assert grade in {0,1,2}
    for did in row['satisfying_doc_ids']:
        assert row['graded_relevance'][did] == 2
    for did in row['violating_doc_ids']:
        assert row['graded_relevance'][did] == 0

assert len(all_meta) == 600, len(all_meta)
assert len(semantic_units) == 200
assert all(c == 3 for c in semantic_counts.values())

stats = {
    'num_corpus_docs': len(ordered_doc_ids),
    'num_semantic_units': len(semantic_units),
    'num_queries': len(all_meta),
    'num_pair_units': sum(1 for u in semantic_units if u.kind == 'pair'),
    'num_single_units': sum(1 for u in semantic_units if u.kind == 'single'),
    'avg_satisfying_docs_per_query': float(np.mean([len(r['satisfying_doc_ids']) for r in all_meta])),
    'avg_violating_docs_per_query': float(np.mean([len(r['violating_doc_ids']) for r in all_meta])),
    'avg_irrelevant_docs_per_query': float(np.mean([len(r['judged_irrelevant_doc_ids']) for r in all_meta])),
    'avg_topical_docs_per_query': float(np.mean([len(r['topical_relevant_doc_ids']) for r in all_meta])),
    'guide_file_used': 'scifact_negation_generation_guide.md',
    'construction_note': 'weakly supervised query-level rejudging from SciFact BEIR corpus and queries; satisfying/violating labels are inferred from abstract-level sentence retrieval and polarity-aware scoring, then exported in the requested query-level schema.',
}
with open(STATS_OUT, 'w') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print('Done.')
print(json.dumps(stats, indent=2, ensure_ascii=False))
