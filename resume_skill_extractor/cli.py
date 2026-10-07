"""Command-line interface: ``python -m resume_skill_extractor resume.pdf [--jd job.txt]``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .extractor import ExtractionResult, SkillExtractor
from .jd_match import JobMatch, match_job
from .readers import SUPPORTED_EXTENSIONS, read_file
from .taxonomy import Taxonomy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="resume-skills",
        description="Extract skills from resumes and optionally score them against a job description.",
    )
    parser.add_argument("resumes", nargs="+", type=Path, help=f"resume files ({', '.join(SUPPORTED_EXTENSIONS)})")
    parser.add_argument("--jd", "--job", dest="job", type=Path, help="job description file to match against")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--min-confidence", type=float, default=0.0, metavar="X", help="hide skills below X (0-1)")
    parser.add_argument("--evidence", action="store_true", help="show the resume lines each skill was found on")
    parser.add_argument("--taxonomy", type=Path, help="custom skills JSON (default: built-in taxonomy)")
    args = parser.parse_args(argv)

    # Resume text can contain characters the Windows console code page cannot print.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

    try:
        extractor = SkillExtractor(Taxonomy.load(args.taxonomy) if args.taxonomy else None)
        job_text = read_file(args.job) if args.job else None
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    reports: list[tuple[Path, ExtractionResult, JobMatch | None]] = []
    failed = False
    for path in args.resumes:
        try:
            result = extractor.extract(read_file(path)).filtered(args.min_confidence)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
            failed = True
            continue
        match = match_job(result, job_text, extractor) if job_text is not None else None
        reports.append((path, result, match))

    if args.json:
        payload = [_as_json(path, result, match) for path, result, match in reports]
        print(json.dumps(payload[0] if len(args.resumes) == 1 and payload else payload, indent=2))
    else:
        for path, result, match in reports:
            print(format_report(path, result, match, show_evidence=args.evidence))
        if args.job and len(reports) > 1:
            print(format_ranking(args.job, reports))
    return 1 if failed else 0


def _as_json(path: Path, result: ExtractionResult, match: JobMatch | None) -> dict:
    data = {"file": str(path), **result.to_dict()}
    if match is not None:
        data["job_match"] = match.to_dict()
    return data


def format_report(
    path: Path, result: ExtractionResult, match: JobMatch | None = None, *, show_evidence: bool = False
) -> str:
    summary = result.to_dict()["summary"]
    out = [
        f"Resume: {path}",
        f"Sections detected: {', '.join(result.sections) or 'none'}",
        f"Found {summary['total']} skills ({summary['high']} high, {summary['medium']} medium, "
        f"{summary['low']} low confidence)",
    ]
    for category, skills in result.by_category().items():
        out.append("")
        out.append(category)
        for skill in skills:
            bar = "#" * round(skill.confidence * 10)
            out.append(
                f"  {skill.name:<30} [{bar:.<10}] {skill.confidence:.2f}  "
                f"x{len(skill.mentions):<3} {', '.join(skill.sections)}"
            )
            if show_evidence:
                out.extend(f"      > {line}" for line in skill.evidence)
    if result.other_skills:
        out += ["", "Also listed (not in taxonomy): " + ", ".join(result.other_skills)]
    if match is not None:
        out += ["", *_format_match(match)]
    out.append("")
    return "\n".join(out)


def _format_match(match: JobMatch) -> list[str]:
    def listed(label: str, items: list[str]) -> str:
        return f"  {label} ({len(items)}): {', '.join(items) or '-'}"

    return [
        "Job match",
        f"  Score: {match.score:.1f}/100   skill coverage {match.skill_coverage:.0%}   "
        f"text similarity {match.text_similarity:.2f}",
        listed("Matched", match.matched),
        listed("Missing required", match.missing_required),
        listed("Missing preferred", match.missing_preferred),
        listed("Extra resume skills", match.extra),
    ]


def format_ranking(job: Path, reports: list[tuple[Path, ExtractionResult, JobMatch | None]]) -> str:
    ranked = sorted(reports, key=lambda r: -r[2].score)
    out = [f"Ranking against {job}"]
    for rank, (path, _, match) in enumerate(ranked, 1):
        out.append(
            f"  {rank}. {match.score:5.1f}  {path.name}  "
            f"(coverage {match.skill_coverage:.0%}, missing: {', '.join(match.missing_required) or 'none'})"
        )
    return "\n".join(out) + "\n"
