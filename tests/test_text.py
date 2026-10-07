from resume_skill_extractor.matcher import PhraseMatcher
from resume_skill_extractor.text import clean_text, phrase_key, tokenize


def test_tokenizer_keeps_symbol_heavy_skill_names_whole():
    surfaces = [t.text for t in tokenize("C++, C#, .NET, Node.js and ASP.NET")]
    assert surfaces == ["C++", ",", "C#", ",", ".NET", ",", "Node.js", "and", "ASP.NET"]


def test_hyphens_and_slashes_separate_words():
    assert phrase_key("machine-learning") == phrase_key("Machine Learning")
    assert phrase_key("CI/CD") == ("ci", "cd")


def test_plurals_normalise_to_singular():
    assert phrase_key("REST APIs") == phrase_key("REST API")
    assert phrase_key("technologies") == ("technology",)
    assert phrase_key("Redis") == ("redis",)
    assert phrase_key("AWS") == ("aws",)


def test_ellipsis_does_not_glue_onto_next_word():
    assert [t.text for t in tokenize("etc...Python")][-1] == "Python"


def test_tokens_know_their_line():
    assert [t.line for t in tokenize("Python\n\nJava")] == [0, 2]


def test_clean_text_normalises_bullets_dashes_and_blank_lines():
    assert clean_text(" Python – 3 yrs\r\n\r\n\r\nJava") == "• Python - 3 yrs\n\nJava"


def test_matcher_prefers_longest_phrase():
    matcher = PhraseMatcher({("react",): "React", ("react", "native"): "React Native"})
    assert matcher.find(["react", "native", "and", "react"]) == [(0, 2, "React Native"), (3, 4, "React")]
