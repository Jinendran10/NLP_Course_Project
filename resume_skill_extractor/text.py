"""Text cleaning and tokenisation shared by every stage of the pipeline.

The same normalisation is applied to resume text and to taxonomy aliases, so
"Machine-Learning", "machine learning" and "Machine Learnings" all reduce to
the same token sequence and therefore match one another.
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass

# Bullet glyphs left behind by PDF/DOCX extraction (including the Symbol-font
# private-use bullets Word emits). They all become "•" so later stages only
# have to recognise one list marker.
_BULLET_RE = re.compile(
    "[•●▪■◦○◆◇►▶"
    "➢➤✓✔∙·]"
)
_DASH_RE = re.compile("[‐-―−]")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_SPACE_RE = re.compile(r"[ \t\f\v]+")

_TOKEN_RE = re.compile(
    r"""
      [^\W\d_]+&[^\W\d_]+                           # R&D, AT&T, W&B
    | (?:(?<![\w.])\.)?[^\W_]+(?:\.[^\W_]+)*[+#]*   # python, node.js, .net, c++, c#
    | [^\w\s]                                       # any other single symbol
    """,
    re.VERBOSE,
)

# Hyphens and slashes are treated like spaces: "scikit-learn" == "scikit learn",
# "CI/CD" == "CI CD". They are dropped from the token stream entirely.
_JOINERS = frozenset("-/")
# Plural acronyms that the "-is" rule below (analysis, redis) would otherwise keep.
_ACRONYM_PLURALS = frozenset({"apis", "kpis", "guis"})


@dataclass(frozen=True, slots=True)
class Token:
    text: str  # surface form exactly as written
    norm: str  # lower-cased, singularised key used for matching
    start: int  # character offsets into the source text
    end: int
    line: int  # 0-based line number


def clean_text(text: str) -> str:
    """Normalise unicode, bullets, dashes and whitespace while keeping line structure."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _BULLET_RE.sub(" • ", text)
    text = _DASH_RE.sub("-", text).translate(_QUOTES)

    lines: list[str] = []
    for raw in text.split("\n"):
        line = _SPACE_RE.sub(" ", raw).strip()
        if line or (lines and lines[-1]):  # keep at most one blank line in a row
            lines.append(line)
    return "\n".join(lines).strip()


def singularize(word: str) -> str:
    """Cheap rule-based singular form; it only has to be consistent, not correct."""
    if len(word) <= 3 or not word.isalpha():
        return word
    if word in _ACRONYM_PLURALS:
        return word[:-1]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def normalize_token(surface: str) -> str:
    return singularize(surface.lower())


def tokenize(text: str) -> list[Token]:
    line_starts = [0] + [m.end() for m in re.finditer("\n", text)]
    tokens = []
    for m in _TOKEN_RE.finditer(text):
        surface = m.group()
        if surface in _JOINERS:
            continue
        line = bisect.bisect_right(line_starts, m.start()) - 1
        tokens.append(Token(surface, normalize_token(surface), m.start(), m.end(), line))
    return tokens


def phrase_key(phrase: str) -> tuple[str, ...]:
    """The normalised token sequence a phrase is matched by."""
    return tuple(t.norm for t in tokenize(phrase))
