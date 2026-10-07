"""Skill taxonomy: canonical skills, their categories and the aliases that refer to them.

The taxonomy is a JSON object mapping category -> list of entries:

    {"name": "Go", "aliases": ["golang"], "strict": ["Go"]}

* ``aliases``         match in any casing ("golang", "GoLang").
* ``case_sensitive``  match only with the exact casing given ("React", not "react").
* ``strict``          exact casing *and* supporting context, for words that are
                      also ordinary English or single letters ("Go", "R", "C").

The canonical name is added as a case-insensitive alias automatically, unless it
is already listed under ``case_sensitive`` or ``strict``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from .text import phrase_key, tokenize

DEFAULT_TAXONOMY = Path(__file__).with_name("data") / "skills.json"


@dataclass(frozen=True)
class Skill:
    name: str
    category: str


@dataclass(frozen=True)
class Alias:
    skill: Skill
    mode: str  # "any" | "case" | "strict"
    forms: frozenset[tuple[str, ...]] = frozenset()  # accepted surface spellings for case/strict


class Taxonomy:
    def __init__(self, data: dict[str, list[dict]]) -> None:
        self.categories: list[str] = []
        self.skills: dict[str, Skill] = {}
        self.aliases: dict[tuple[str, ...], Alias] = {}
        for category, entries in data.items():
            self.categories.append(category)
            for entry in entries:
                self._add(entry, category)

    @classmethod
    def load(cls, path: str | PathLike[str] | None = None) -> Taxonomy:
        with open(path or DEFAULT_TAXONOMY, encoding="utf-8") as f:
            return cls(json.load(f))

    def __len__(self) -> int:
        return len(self.skills)

    def _add(self, entry: dict, category: str) -> None:
        name = entry["name"]
        if name in self.skills:
            raise ValueError(f"skill {name!r} is defined twice")
        skill = self.skills[name] = Skill(name, category)

        own: dict[tuple[str, ...], Alias] = {}
        for mode, field in (("case", "case_sensitive"), ("strict", "strict")):
            for form in entry.get(field, []):
                key = phrase_key(form)
                previous = own.get(key)
                if previous is not None and previous.mode != mode:
                    raise ValueError(f"{name!r}: {form!r} is listed as both case-sensitive and strict")
                surface = tuple(t.text for t in tokenize(form))
                forms = (previous.forms if previous else frozenset()) | {surface}
                own[key] = Alias(skill, mode, forms)

        for alias in entry.get("aliases", []):
            key = phrase_key(alias)
            if key in own and own[key].mode != "any":
                raise ValueError(f"{name!r}: alias {alias!r} is also listed as case-sensitive/strict")
            own[key] = Alias(skill, "any")
        own.setdefault(phrase_key(name), Alias(skill, "any"))

        for key, alias in own.items():
            if not key:
                raise ValueError(f"{name!r} has an empty alias")
            other = self.aliases.get(key)
            if other is not None:
                raise ValueError(
                    f"alias {' '.join(key)!r} is claimed by both {other.skill.name!r} and {name!r}"
                )
            self.aliases[key] = alias
