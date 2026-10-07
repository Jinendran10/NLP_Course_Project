"""Score an extracted resume against a job description."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass

from .extractor import ExtractionResult, SkillExtractor
from .text import normalize_token, tokenize

PREFERRED_WEIGHT = 0.5  # a "nice to have" skill counts half as much as a requirement
SKILL_WEIGHT = 0.8  # share of the final score from skill coverage; the rest is text similarity

_PREFERRED_RE = re.compile(
    r"\b(?:preferred|nice[\s-]to[\s-]have|good[\s-]to[\s-]have|bonus|desirable|desired|"
    r"a plus|optional|ideally)\b",
    re.IGNORECASE,
)
_REQUIRED_RE = re.compile(
    r"\b(?:requirements?|required|qualifications?|must[\s-]haves?|responsibilities|"
    r"what you(?:'ll)? (?:need|bring|do)|who you are|you have|skills|about you|minimum)\b",
    re.IGNORECASE,
)
STOPWORDS = frozenset(
    normalize_token(w)
    for w in """a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each etc few for
    from further had has have having he her here hers him his how i if in including into is it its
    just me more most my no nor not now of off on once only or other our out over own same she
    should so some such than that the their them then there these they this those through to too
    under until up very was we were what when where which while who whom why will with within would
    you your experience year team role ability strong work working using use used""".split()
)


@dataclass
class JobMatch:
    score: float  # 0-100
    skill_coverage: float  # 0-1, weighted share of the job's skills the resume has
    text_similarity: float  # 0-1, cosine similarity of the two texts' content words
    required: list[str]
    preferred: list[str]
    matched: list[str]
    missing_required: list[str]
    missing_preferred: list[str]
    extra: list[str]  # resume skills the job does not mention

    def to_dict(self) -> dict:
        return asdict(self)


def match_job(resume: ExtractionResult, job_text: str, extractor: SkillExtractor) -> JobMatch:
    job = extractor.extract(job_text, detect_name=False)
    labels = requirement_labels(job.text)

    required, preferred = [], []
    for skill in sorted(job.skills, key=lambda s: min(m.start for m in s.mentions)):
        if any(labels[m.line] == "required" for m in skill.mentions):
            required.append(skill.name)
        else:
            preferred.append(skill.name)

    have = set(resume.names())
    matched_required = [s for s in required if s in have]
    matched_preferred = [s for s in preferred if s in have]
    wanted = set(required) | set(preferred)

    total = len(required) + PREFERRED_WEIGHT * len(preferred)
    coverage = (len(matched_required) + PREFERRED_WEIGHT * len(matched_preferred)) / total if total else 0.0
    similarity = text_similarity(resume.text, job.text)
    # Without any recognisable skills in the job ad, only the text similarity is meaningful.
    score = 100 * (SKILL_WEIGHT * coverage + (1 - SKILL_WEIGHT) * similarity) if total else 100 * similarity

    return JobMatch(
        score=round(score, 1),
        skill_coverage=round(coverage, 3),
        text_similarity=similarity,
        required=required,
        preferred=preferred,
        matched=matched_required + matched_preferred,
        missing_required=[s for s in required if s not in have],
        missing_preferred=[s for s in preferred if s not in have],
        extra=[s for s in resume.names() if s not in wanted],
    )


def requirement_labels(text: str) -> list[str]:
    """Label each line of a job ad "required" or "preferred".

    Headings such as "Nice to have:" switch the mode for the lines below them;
    a line like "Terraform is a plus" is preferred on its own.
    """
    mode, labels = "required", []
    for line in text.split("\n"):
        stripped = line.strip()
        words = stripped.split()
        is_heading = bool(words) and stripped[0] not in "•*-" and (stripped.endswith(":") or len(words) <= 4)
        if is_heading and _PREFERRED_RE.search(stripped):
            mode = "preferred"
        elif is_heading and _REQUIRED_RE.search(stripped):
            mode = "required"
        if mode == "preferred" or _PREFERRED_RE.search(stripped):
            labels.append("preferred")
        else:
            labels.append("required")
    return labels


def text_similarity(a: str, b: str) -> float:
    """Cosine similarity of sublinear term-frequency vectors over content words."""
    va, vb = _term_vector(a), _term_vector(b)
    if not va or not vb:
        return 0.0
    dot = sum(weight * vb[term] for term, weight in va.items() if term in vb)
    norm = math.sqrt(sum(w * w for w in va.values())) * math.sqrt(sum(w * w for w in vb.values()))
    return round(dot / norm, 3)


def _term_vector(text: str) -> dict[str, float]:
    counts = Counter(
        t.norm for t in tokenize(text) if t.norm.isalpha() and len(t.norm) > 2 and t.norm not in STOPWORDS
    )
    return {term: 1 + math.log(count) for term, count in counts.items()}
