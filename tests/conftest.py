from pathlib import Path

import pytest

from resume_skill_extractor import SkillExtractor

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture(scope="session")
def extractor() -> SkillExtractor:
    return SkillExtractor()


@pytest.fixture(scope="session")
def samples() -> Path:
    return SAMPLES
