def test_sample_resume(extractor, samples):
    result = extractor.extract_file(samples / "sample_resume.txt")
    names = set(result.names())
    assert {"Python", "SQL", "Go", "R", "Apache Spark", "Apache Kafka", "Snowflake", "AWS Lambda"} <= names
    assert "C" not in names  # "grade C in Organic Chemistry"
    assert result.other_skills == ["Great Expectations"]
    assert result.sections == ["summary", "skills", "experience", "projects", "education", "certifications"]


def test_ambiguous_words_without_context_are_ignored(extractor):
    text = "Experience\nLed the Go to market plan. Received grade C in Physics. Helped the team react quickly."
    assert extractor.extract(text).names() == []


def test_ambiguous_words_with_context_are_accepted(extractor):
    result = extractor.extract("Projects\nWrote a CLI tool in Go. Analysis in R and Python.")
    assert {"Go", "R", "Python"} <= set(result.names())


def test_case_sensitive_aliases(extractor):
    assert "React" in extractor.extract("Experience\nBuilt dashboards in React.").names()
    assert "React" not in extractor.extract("Experience\nTrained staff to react to outages.").names()
    assert "React" in extractor.extract("Skills\nreact, redux").names()  # a skills list relaxes casing


def test_confidence_reflects_where_a_skill_appears(extractor):
    result = extractor.extract("Summary\nInterested in Docker.\nSkills\nPython\nExperience\nShipped Python services.")
    python, docker = result.get("Python"), result.get("Docker")
    assert python.confidence > docker.confidence
    assert python.level == "high"
    assert docker.level == "low"
    assert python.sections == ["skills", "experience"]


def test_name_line_is_not_mined_for_skills(extractor):
    assert extractor.extract("Julia Ruby\nSkills\nSQL").names() == ["SQL"]


def test_unknown_items_in_skills_list_are_reported(extractor):
    result = extractor.extract("Skills\nLanguages: Python, Advanced Foobar, 3 years\nTools: Docker, etc.")
    assert result.other_skills == ["Foobar"]
    assert {"Python", "Docker"} <= set(result.names())


def test_min_confidence_filter(extractor):
    result = extractor.extract("Summary\nDocker\nSkills\nPython").filtered(0.5)
    assert result.names() == ["Python"]


def test_result_serialises(extractor):
    data = extractor.extract("Skills\nPython, Docker").to_dict()
    assert data["summary"] == {"total": 2, "high": 0, "medium": 2, "low": 0}
    assert data["by_category"] == {"Programming Languages": ["Python"], "Cloud & DevOps": ["Docker"]}
    assert data["skills"][0]["evidence"] == ["Python, Docker"]
