"""The extraction pipeline: clean -> segment -> tokenise -> match -> disambiguate -> score."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from os import PathLike

from .matcher import PhraseMatcher
from .readers import read_file
from .sections import Line, is_heading, split_sections
from .taxonomy import Alias, Taxonomy
from .text import Token, clean_text, normalize_token, tokenize

# How much a single mention counts as evidence, by the section it appears in.
# A skill listed under "Skills" *and* used in "Experience" ends up near the top.
SECTION_WEIGHTS = {
    "skills": 0.60,
    "experience": 0.55,
    "projects": 0.55,
    "certifications": 0.45,
    "achievements": 0.40,
    "publications": 0.40,
    "summary": 0.35,
    "education": 0.30,
    "other": 0.30,
}
STRICT_PENALTY = 0.75  # "Go"/"R" in running text is weaker evidence than in a skills list
HIGH_CONFIDENCE, MEDIUM_CONFIDENCE = 0.75, 0.50
SOFT_SKILLS = "Soft Skills"

# Context needed to accept a strict alias such as "Go", "R" or "C".
CUE_WINDOW = 3  # tokens either side searched for a cue word
NEIGHBOUR_GAP = 3  # max tokens between a strict mention and a neighbouring skill
CUE_WORDS = frozenset(
    normalize_token(w)
    for w in """programming language framework library developer development develop developed
    developing code coding script scripting proficient proficiency experience experienced skilled
    using stack tool tech technology built build wrote written implemented programmed engineer api
    app application service backend frontend database server""".split()
)
# Tokens allowed between a strict mention and a neighbouring skill, i.e. the two
# appear in the same list: "Python, R and SQL", "(Go / Rust)".
LIST_GLUE = frozenset({",", "&", "and", "or", "(", ")", "•", "|", "+", ";"})

_LIST_SPLIT_RE = re.compile(r"[,;|•()\[\]{}]|\s+(?:and|&)\s+")
_SUB_LABEL_RE = re.compile(r"^[^:,]{1,40}:\s*")
_QUALIFIER_RE = re.compile(
    r"^(?:advanced|intermediate|basic|beginner|expert|strong|good|excellent|"
    r"(?:proficient|experienced|skilled) (?:in|with)|familiar with|(?:working )?knowledge of|exposure to)\s+",
    re.IGNORECASE,
)
_NOT_SKILLS = frozenset(
    "etc others other more various misc tools languages frameworks libraries databases "
    "technologies skills platforms concepts".split()
)


@dataclass(frozen=True)
class Mention:
    surface: str  # the text as written in the resume
    section: str
    line: int
    start: int
    end: int
    weight: float


@dataclass
class ExtractedSkill:
    name: str
    category: str
    mentions: list[Mention]
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    @property
    def level(self) -> str:
        if self.confidence >= HIGH_CONFIDENCE:
            return "high"
        return "medium" if self.confidence >= MEDIUM_CONFIDENCE else "low"

    @property
    def sections(self) -> list[str]:
        return list(dict.fromkeys(m.section for m in self.mentions))

    @property
    def surfaces(self) -> list[str]:
        return list(dict.fromkeys(m.surface for m in self.mentions))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "confidence": self.confidence,
            "level": self.level,
            "mentions": len(self.mentions),
            "sections": self.sections,
            "matched_as": self.surfaces,
            "evidence": self.evidence,
        }


@dataclass
class ExtractionResult:
    text: str  # the cleaned text the offsets refer to
    skills: list[ExtractedSkill]  # sorted by confidence, highest first
    other_skills: list[str]  # items listed under a skills heading but unknown to the taxonomy
    sections: list[str]
    categories: list[str]  # taxonomy category order, used for grouping

    def names(self) -> list[str]:
        return [s.name for s in self.skills]

    def get(self, name: str) -> ExtractedSkill | None:
        return next((s for s in self.skills if s.name == name), None)

    def filtered(self, min_confidence: float) -> ExtractionResult:
        return replace(self, skills=[s for s in self.skills if s.confidence >= min_confidence])

    def by_category(self) -> dict[str, list[ExtractedSkill]]:
        groups: dict[str, list[ExtractedSkill]] = {c: [] for c in self.categories}
        for skill in self.skills:
            groups.setdefault(skill.category, []).append(skill)
        return {c: skills for c, skills in groups.items() if skills}

    def to_dict(self) -> dict:
        levels = [s.level for s in self.skills]
        return {
            "skills": [s.to_dict() for s in self.skills],
            "by_category": {c: [s.name for s in group] for c, group in self.by_category().items()},
            "other_skills": self.other_skills,
            "sections": self.sections,
            "summary": {
                "total": len(self.skills),
                "high": levels.count("high"),
                "medium": levels.count("medium"),
                "low": levels.count("low"),
            },
        }


class SkillExtractor:
    def __init__(self, taxonomy: Taxonomy | None = None) -> None:
        self.taxonomy = taxonomy or Taxonomy.load()
        self._matcher: PhraseMatcher[Alias] = PhraseMatcher(self.taxonomy.aliases)

    def extract_file(self, path: str | PathLike[str]) -> ExtractionResult:
        return self.extract(read_file(path))

    def extract(self, text: str, *, detect_name: bool = True) -> ExtractionResult:
        """Extract skills from resume text.

        ``detect_name`` skips a leading "Firstname Lastname" line so names like
        "Ruby" or "Julia" are not read as skills; turn it off for job ads.
        """
        text = clean_text(text)
        lines = split_sections(text, detect_name=detect_name)
        tokens = [t for t in tokenize(text) if t.start >= lines[t.line].body_start]
        candidates = self._matcher.find([t.norm for t in tokens])
        accepted = self._disambiguate(candidates, tokens, lines)
        return ExtractionResult(
            text=text,
            skills=self._aggregate(accepted, tokens, lines, text),
            other_skills=self._other_skills(lines),
            sections=list(dict.fromkeys(line.heading for line in lines if line.heading)),
            categories=self.taxonomy.categories,
        )

    # -- disambiguation ------------------------------------------------------

    def _disambiguate(
        self, candidates: list[tuple[int, int, Alias]], tokens: list[Token], lines: list[Line]
    ) -> list[tuple[int, int, Alias]]:
        accepted, strict = [], []
        for start, end, alias in candidates:
            if alias.mode == "any":
                accepted.append((start, end, alias))
            elif _casing_matches(alias, tokens[start:end], lines[tokens[start].line]):
                (strict if alias.mode == "strict" else accepted).append((start, end, alias))

        # Unambiguous technical skills act as anchors that vouch for strict ones.
        anchors = [(s, e) for s, e, a in accepted if a.skill.category != SOFT_SKILLS]
        for start, end, alias in strict:
            if _has_support(start, end, tokens, lines, anchors):
                accepted.append((start, end, alias))
        accepted.sort(key=lambda match: match[0])
        return accepted

    # -- scoring -------------------------------------------------------------

    def _aggregate(
        self, accepted: list[tuple[int, int, Alias]], tokens: list[Token], lines: list[Line], text: str
    ) -> list[ExtractedSkill]:
        found: dict[str, ExtractedSkill] = {}
        for start, end, alias in accepted:
            first, last = tokens[start], tokens[end - 1]
            section = lines[first.line].section
            weight = SECTION_WEIGHTS.get(section, SECTION_WEIGHTS["other"])
            if alias.mode == "strict" and section != "skills":
                weight *= STRICT_PENALTY
            skill = found.setdefault(
                alias.skill.name, ExtractedSkill(alias.skill.name, alias.skill.category, [])
            )
            skill.mentions.append(
                Mention(text[first.start : last.end], section, first.line, first.start, last.end, weight)
            )

        for skill in found.values():
            skill.confidence = _confidence(skill.mentions)
            skill.evidence = _evidence(skill.mentions, lines)
        return sorted(found.values(), key=lambda s: (-s.confidence, -len(s.mentions), s.name.lower()))

    # -- open-vocabulary skills ---------------------------------------------

    def _other_skills(self, lines: list[Line]) -> list[str]:
        """Items in a skills list that the taxonomy does not know, e.g. "Great Expectations"."""
        found: dict[str, str] = {}
        for line in lines:
            if line.section != "skills":
                continue
            body = _SUB_LABEL_RE.sub("", line.body, count=1)
            parts = _LIST_SPLIT_RE.split(body)
            if len(parts) == 1 and len(body.split()) > 5:
                continue  # a sentence, not a list
            for part in parts:
                item = _QUALIFIER_RE.sub("", part.strip(" .:-*\t")).strip(" .:-*\t")
                if self._is_unknown_skill(item):
                    found.setdefault(item.lower(), item)
        return list(found.values())

    def _is_unknown_skill(self, item: str) -> bool:
        if not item or len(item) > 40 or len(item.split()) > 4:
            return False
        if item[0].isdigit() or not any(c.isalpha() for c in item) or item.lower() in _NOT_SKILLS:
            return False
        if is_heading(item):
            return False
        # Anything the taxonomy recognises, accepted or not, is not "unknown".
        return not self._matcher.find([t.norm for t in tokenize(item)])


def _casing_matches(alias: Alias, span: list[Token], line: Line) -> bool:
    surface = tuple(t.text for t in span)
    if surface in alias.forms:
        return True
    # Inside a skills list "react" clearly means React; elsewhere only the
    # exact casing counts. Strict aliases ("Go", "C") always need exact casing.
    if alias.mode == "case" and line.section == "skills":
        return True
    # In an ALL-CAPS line casing carries no information.
    letters = [c for c in line.text if c.isalpha()]
    if len(letters) >= 4 and all(c.isupper() for c in letters):
        upper = tuple(s.upper() for s in surface)
        return any(tuple(f.upper() for f in form) == upper for form in alias.forms)
    return False


def _has_support(
    start: int, end: int, tokens: list[Token], lines: list[Line], anchors: list[tuple[int, int]]
) -> bool:
    line = tokens[start].line
    if lines[line].section == "skills":
        return True
    window = tokens[max(0, start - CUE_WINDOW) : start] + tokens[end : end + CUE_WINDOW]
    if any(t.line == line and t.norm in CUE_WORDS for t in window):
        return True
    for a_start, a_end in anchors:
        between = tokens[a_end:start] if a_end <= start else tokens[end:a_start]
        if (
            tokens[a_start].line == line
            and len(between) <= NEIGHBOUR_GAP
            and all(t.norm in LIST_GLUE for t in between)
        ):
            return True
    return False


def _confidence(mentions: list[Mention]) -> float:
    """Noisy-OR over lines: each line is independent evidence, repeats on a line count once."""
    best: dict[int, float] = {}
    for m in mentions:
        best[m.line] = max(best.get(m.line, 0.0), m.weight)
    miss = 1.0
    for weight in best.values():
        miss *= 1.0 - weight
    return round(min(1.0 - miss, 0.99), 2)


def _evidence(mentions: list[Mention], lines: list[Line], limit: int = 3) -> list[str]:
    snippets: list[str] = []
    for m in mentions:
        snippet = lines[m.line].text.strip(" •")
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        if snippet not in snippets:
            snippets.append(snippet)
        if len(snippets) == limit:
            break
    return snippets
