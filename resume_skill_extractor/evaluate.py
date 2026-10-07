"""Precision / recall / F1 against a hand-labelled JSONL file.

Each line: {"id": "...", "text": "...", "skills": ["Python", ...]} with canonical
skill names from the taxonomy. Run: ``python -m resume_skill_extractor.evaluate``.
"""

from __future__ import annotations

import argparse
import json
from os import PathLike
from pathlib import Path

from .extractor import SkillExtractor

DEFAULT_GOLD = Path(__file__).resolve().parent.parent / "eval" / "gold.jsonl"


def evaluate(
    path: str | PathLike[str] = DEFAULT_GOLD,
    extractor: SkillExtractor | None = None,
    min_confidence: float = 0.0,
) -> dict:
    extractor = extractor or SkillExtractor()
    tp = fp = fn = 0
    examples = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        example = json.loads(line)
        gold = set(example["skills"])
        unknown = gold - extractor.taxonomy.skills.keys()
        if unknown:
            raise ValueError(f"line {number}: gold skills not in taxonomy: {sorted(unknown)}")
        predicted = set(extractor.extract(example["text"]).filtered(min_confidence).names())
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        examples.append(
            {
                "id": example.get("id", str(number)),
                "false_positives": sorted(predicted - gold),
                "false_negatives": sorted(gold - predicted),
            }
        )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "examples": examples,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate skill extraction against labelled examples.")
    parser.add_argument("gold", nargs="?", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args(argv)

    report = evaluate(args.gold, min_confidence=args.min_confidence)
    for ex in report["examples"]:
        status = "ok" if not ex["false_positives"] and not ex["false_negatives"] else "errors"
        print(f"{ex['id']:<28} {status}")
        if ex["false_positives"]:
            print(f"    false positives: {', '.join(ex['false_positives'])}")
        if ex["false_negatives"]:
            print(f"    false negatives: {', '.join(ex['false_negatives'])}")
    print(
        f"\nprecision {report['precision']:.3f}  recall {report['recall']:.3f}  f1 {report['f1']:.3f}  "
        f"(tp={report['true_positives']} fp={report['false_positives']} fn={report['false_negatives']})"
    )


if __name__ == "__main__":
    main()
