import pytest

from resume_skill_extractor.sections import split_sections
from resume_skill_extractor.taxonomy import Taxonomy


def test_default_taxonomy_loads_without_conflicts():
    taxonomy = Taxonomy.load()
    assert len(taxonomy) > 300
    assert taxonomy.skills["Python"].category == "Programming Languages"


def test_alias_claimed_by_two_skills_is_rejected():
    with pytest.raises(ValueError, match="claimed by both"):
        Taxonomy({"X": [{"name": "Foo", "aliases": ["shared"]}, {"name": "Bar", "aliases": ["Shared"]}]})


def test_strict_name_is_not_also_a_plain_alias():
    taxonomy = Taxonomy({"Languages": [{"name": "Go", "aliases": ["golang"], "strict": ["Go"]}]})
    assert taxonomy.aliases[("go",)].mode == "strict"
    assert taxonomy.aliases[("golang",)].mode == "any"


def test_detects_common_headings_and_name_line():
    text = "Jane Doe\nSUMMARY\nBuilt things.\nTechnical Skills:\nPython\nWork Experience\nEngineer\nE D U C A T I O N\nBSc"
    sections = [line.section for line in split_sections(text, detect_name=True)]
    assert sections == [
        "name", "summary", "summary", "skills", "skills", "experience", "experience", "education", "education",
    ]


def test_inline_heading_labels_only_its_own_line():
    lines = split_sections("Experience\nTech Stack: React, Node.js\nShipped the checkout flow")
    assert [line.section for line in lines] == ["experience", "skills", "experience"]
    assert lines[1].body == "React, Node.js"


def test_sentence_ending_in_skills_is_not_a_heading():
    lines = split_sections("Summary\nExcellent communication skills.")
    assert lines[1].section == "summary"
    assert lines[1].heading is None
