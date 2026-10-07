import pytest

from resume_skill_extractor import match_job
from resume_skill_extractor.jd_match import requirement_labels, text_similarity


def test_required_and_preferred_skills(extractor):
    resume = extractor.extract("Skills\nPython, Docker, Scala")
    job = "Requirements:\n• Python and Kubernetes\nNice to have:\n• Scala or Java"
    match = match_job(resume, job, extractor)
    assert match.required == ["Python", "Kubernetes"]
    assert match.preferred == ["Scala", "Java"]
    assert match.matched == ["Python", "Scala"]
    assert match.missing_required == ["Kubernetes"]
    assert match.missing_preferred == ["Java"]
    assert match.extra == ["Docker"]
    assert match.skill_coverage == pytest.approx((1 + 0.5) / (2 + 0.5 * 2), abs=1e-3)


def test_inline_plus_marks_a_line_preferred():
    labels = requirement_labels("Responsibilities:\n• Build APIs in Python\n• Terraform experience is a plus")
    assert labels == ["required", "required", "preferred"]


def test_text_similarity_bounds():
    assert text_similarity("python data pipelines", "python data pipelines") == pytest.approx(1.0)
    assert text_similarity("python data pipelines", "") == 0.0


def test_sample_match(extractor, samples):
    resume = extractor.extract_file(samples / "sample_resume.txt")
    match = match_job(resume, (samples / "sample_jd.txt").read_text(encoding="utf-8"), extractor)
    assert match.missing_required == ["BigQuery", "Infrastructure as Code"]
    assert match.missing_preferred == ["Java"]
    assert 60 < match.score < 90
