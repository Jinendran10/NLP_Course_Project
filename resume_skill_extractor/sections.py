"""Split a resume into labelled sections (summary, skills, experience, ...).

Every line gets a section label. A line that is only a heading ("TECHNICAL
SKILLS", "Work Experience:") switches the current section. A line that starts
with a heading label ("Tech Stack: React, Node.js") is labelled with that
section for itself only, because such labels often appear inside project or
job entries. The heading text itself is never treated as content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SECTION_HEADINGS: dict[str, tuple[str, ...]] = {
    "summary": (
        "summary", "professional summary", "career summary", "profile", "professional profile",
        "profile summary", "about", "about me", "objective", "career objective", "overview",
        "executive summary", "personal statement", "summary of qualifications",
    ),
    "skills": (
        "skills", "technical skills", "key skills", "core skills", "core competencies",
        "competencies", "skills and abilities", "areas of expertise", "expertise", "technologies",
        "tech stack", "technical proficiencies", "skill set", "skillset", "tools and technologies",
        "technical expertise", "it skills", "computer skills", "programming skills",
        "skills summary", "relevant skills", "skills and tools", "technical skills and tools",
        "tools", "technical summary", "proficiencies", "technology stack",
    ),
    "experience": (
        "experience", "work experience", "professional experience", "employment history",
        "work history", "employment", "career history", "relevant experience", "internships",
        "internship", "internship experience", "industry experience", "research experience",
        "volunteer experience", "professional background", "positions held",
    ),
    "projects": (
        "projects", "personal projects", "academic projects", "key projects", "project experience",
        "selected projects", "side projects", "project work", "notable projects", "open source",
        "open source contributions",
    ),
    "education": (
        "education", "academic background", "educational background", "academics",
        "academic qualifications", "qualifications", "education and training", "coursework",
        "relevant coursework", "academic details", "educational qualifications",
    ),
    "certifications": (
        "certifications", "certification", "certificates", "licenses and certifications",
        "certifications and licenses", "certifications and training", "training", "courses",
        "professional development", "licenses",
    ),
    "achievements": (
        "achievements", "awards", "honors", "honours", "awards and honors", "awards and honours",
        "awards and achievements", "accomplishments",
    ),
    "publications": ("publications", "papers", "research", "research papers", "patents"),
    "other": (
        "interests", "hobbies", "hobbies and interests", "activities", "extracurricular activities",
        "extra curricular activities", "volunteering", "references", "personal details",
        "personal information", "declaration", "languages known", "spoken languages",
        "language proficiency", "contact", "contact information",
    ),
}

# Short Title Case headings ending in one of these words are recognised even if
# not listed above ("Cloud Technologies", "Data Science Skills").
_SUFFIX_SECTIONS = {
    "skills": "skills", "skill": "skills", "competencies": "skills", "technologies": "skills",
    "expertise": "skills", "proficiencies": "skills", "experience": "experience",
    "projects": "projects", "certifications": "certifications", "education": "education",
    "awards": "achievements", "achievements": "achievements", "publications": "publications",
}

_LOOKUP = {alias: section for section, aliases in SECTION_HEADINGS.items() for alias in aliases}
_COMPACT_LOOKUP = {alias.replace(" ", ""): section for alias, section in _LOOKUP.items()}
_SMALL_WORDS = {"and", "of", "the", "in", "for", "to"}
_INLINE_RE = re.compile(r"^[\s•*\-]*([A-Za-z][A-Za-z &/]{1,40}?)\s*:\s*(\S.*)$")
_NAME_WORD_RE = re.compile(r"^[A-Z][A-Za-z'.\-]*$")


@dataclass(frozen=True, slots=True)
class Line:
    number: int
    text: str
    start: int  # character offset of the line in the document
    section: str
    body_start: int  # offset where content begins (after any heading label)
    heading: str | None = None  # section this line declares, if it is a heading

    @property
    def body(self) -> str:
        return self.text[self.body_start - self.start :]


def split_sections(text: str, *, detect_name: bool = False) -> list[Line]:
    """Label each line of ``text`` with the section it belongs to.

    With ``detect_name`` a first line that looks like a person's name
    ("Ruby Chen") is labelled "name" so it is never mined for skills.
    """
    lines: list[Line] = []
    current, offset, seen_content = "other", 0, False
    for number, raw in enumerate(text.split("\n")):
        end = offset + len(raw)
        if (section := _full_heading(raw)) is not None:
            current = section
            line = Line(number, raw, offset, section, end, section)
        elif (inline := _inline_heading(raw)) is not None:
            section, column = inline
            line = Line(number, raw, offset, section, offset + column, section)
        elif detect_name and not seen_content and _looks_like_name(raw):
            line = Line(number, raw, offset, "name", end)
        else:
            line = Line(number, raw, offset, current, offset)
        seen_content = seen_content or bool(raw.strip())
        lines.append(line)
        offset = end + 1
    return lines


def is_heading(text: str) -> bool:
    return _full_heading(text) is not None


def _normalise_heading(text: str) -> str:
    text = text.lower().replace("&", " and ")
    words = re.sub(r"[^a-z ]+", " ", text).split()
    if len(words) >= 3 and all(len(w) == 1 for w in words):  # "S K I L L S"
        words = ["".join(words)]
    return " ".join(words)


def _lookup(label: str) -> str | None:
    return _LOOKUP.get(label) or _COMPACT_LOOKUP.get(label.replace(" ", ""))


def _is_title_like(text: str) -> bool:
    words = [w for w in re.findall(r"[A-Za-z]+", text) if w.lower() not in _SMALL_WORDS]
    return bool(words) and all(w[0].isupper() for w in words)


def _full_heading(line: str) -> str | None:
    stripped = line.strip().strip("•*#=_-|").strip()
    has_colon = stripped.endswith(":")
    stripped = stripped.rstrip(":").strip()
    if not stripped or len(stripped) > 50 or stripped.endswith("."):
        return None
    label = _normalise_heading(stripped)
    if not label:
        return None
    if section := _lookup(label):
        return section
    words = label.split()
    if len(words) <= 4 and words[-1] in _SUFFIX_SECTIONS and (has_colon or _is_title_like(stripped)):
        return _SUFFIX_SECTIONS[words[-1]]
    return None


def _inline_heading(line: str) -> tuple[str, int] | None:
    m = _INLINE_RE.match(line)
    if not m:
        return None
    label = _normalise_heading(m.group(1))
    section = _lookup(label)
    if section is None and label and label.split()[-1] in _SUFFIX_SECTIONS and len(label.split()) <= 4:
        section = _SUFFIX_SECTIONS[label.split()[-1]]
    return (section, m.start(2)) if section else None


def _looks_like_name(line: str) -> bool:
    words = line.split()
    return 2 <= len(words) <= 4 and all(_NAME_WORD_RE.match(w) for w in words)
