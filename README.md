# Resume Skill Extractor

Extracts skills from resumes (PDF, DOCX, TXT), groups them by category, scores how strongly the resume supports each one, and optionally compares the resume against a job description.

```
$ python -m resume_skill_extractor samples/sample_resume.txt --jd samples/sample_jd.txt

Resume: samples/sample_resume.txt
Sections detected: summary, skills, experience, projects, education, certifications
Found 50 skills (9 high, 37 medium, 4 low confidence)

Programming Languages
  Python                         [#########.] 0.92  x3   skills, experience
  SQL                            [########..] 0.82  x2   skills, experience
  Go                             [########..] 0.77  x2   skills, experience
  ...
Also listed (not in taxonomy): Great Expectations

Job match
  Score: 76.1/100   skill coverage 84%   text similarity 0.43
  Missing required (2): BigQuery, Infrastructure as Code
  Missing preferred (1): Java
```

## How it works

The extractor matches phrases against a skill taxonomy. It does not use a trained model, so every decision can be explained. The pipeline has these stages:

1. **Text extraction** ([readers.py](resume_skill_extractor/readers.py)). PyMuPDF reads PDFs (pypdf is the fallback) and python-docx reads DOCX files, including table cells. Scanned PDFs have no text layer and are rejected with a message.
2. **Cleaning** ([text.py](resume_skill_extractor/text.py)). Unicode is NFKC-normalised, so ligatures and non-breaking spaces become plain text. The many bullet glyphs become one `•`, and dashes and quotes become ASCII.
3. **Section segmentation** ([sections.py](resume_skill_extractor/sections.py)). Each line is labelled `summary`, `skills`, `experience`, `projects`, `education`, `certifications`, and so on, using a list of about 120 heading variants. It also handles:
   - Title Case headings that end in a known word, such as "Cloud Technologies".
   - Spaced-out headings such as `S K I L L S`.
   - Inline labels such as `Tech Stack: React, Node.js`.

   A first line that looks like a person's name is labelled `name` and skipped, so a candidate called *Ruby* or *Julia* is not read as a skill.
4. **Tokenisation and normalisation**. A regex tokenizer keeps names like `C++`, `C#`, `.NET`, `Node.js` and `R&D` whole. Hyphens and slashes are treated as spaces, so `machine-learning` = `machine learning` and `CI/CD` = `CI CD`. Tokens are lower-cased and singularised with simple rules, so `REST APIs` = `REST API`. Aliases in the taxonomy go through exactly the same normalisation, so any spelling variant matches.
5. **Phrase matching** ([matcher.py](resume_skill_extractor/matcher.py)). A token trie built from the roughly 990 aliases is scanned left to right with longest-match-wins, so `React Native` is not also counted as `React`.
6. **Disambiguation** ([extractor.py](resume_skill_extractor/extractor.py)). Some skill names are also ordinary words or single letters. The taxonomy marks these aliases at one of three levels:
   - `any`: any casing matches (`python`, `PYTHON`).
   - `case_sensitive`: only the exact casing matches (`React` the library, not "react to incidents"; `REST`, not "rest").
   - `strict`: exact casing **and** supporting context (`Go`, `R`, `C`, `Spark`, `Excel`, `Swift`, `Oracle`, ...). Context means any one of:
     - the line is in a skills section;
     - a cue word is within 3 tokens (*programming, using, wrote, developer, api, ...*);
     - the term sits in a list next to an unambiguous technical skill ("R and Python", "C/C++").

   So "Led the Go to market plan" and "grade C in Chemistry" are ignored, while "REST APIs in Go and Python" is accepted. Casing rules relax inside a skills list and on ALL-CAPS lines.
7. **Confidence scoring**. Each mention gets a weight from its section: skills 0.60, experience/projects 0.55, certifications 0.45, summary 0.35, education 0.30. A strict mention in running text is weighted ×0.75. The weights are combined with a noisy-OR (`1 − Π(1 − wᵢ)`), counting each line once. As a result:
   - A skill that is listed **and** used in experience scores high (≥ 0.75).
   - A skill only named in the summary scores low.

   Each skill also carries up to three evidence lines.
8. **Open-vocabulary skills**. Items in a skills list that the taxonomy does not know (for example "Great Expectations") are reported separately as `other_skills`. To find them, the extractor:
   - splits list lines on `, ; | • ( )`;
   - drops sub-labels ("Languages:") and qualifiers ("Advanced", "familiar with");
   - drops anything that looks like a sentence or a duration.
9. **Job matching** ([jd_match.py](resume_skill_extractor/jd_match.py)). The job ad goes through the same extractor. Each skill in it is labelled *required* or *preferred*:
   - it is preferred when it sits under a heading such as "Nice to have:" or "Preferred", or on a line saying "...is a plus";
   - otherwise it is required.

   The score is `100 × (0.8 × weighted skill coverage + 0.2 × text similarity)`:
   - weighted skill coverage counts a preferred skill as 0.5 and a required skill as 1;
   - text similarity is the cosine of sublinear term-frequency vectors over content words.

## Setup

```bash
pip install -r requirements.txt        # pymupdf, python-docx, flask, pytest
```

Python 3.10+.

## Usage

**Command line**

```bash
python -m resume_skill_extractor resume.pdf
python -m resume_skill_extractor resume.pdf --jd job.txt --evidence      # show the lines each skill came from
python -m resume_skill_extractor a.pdf b.docx c.pdf --jd job.txt         # also ranks the candidates
python -m resume_skill_extractor resume.pdf --json --min-confidence 0.5
python -m resume_skill_extractor resume.pdf --taxonomy my_skills.json
```

**Web app**

```bash
python -m resume_skill_extractor.web          # http://127.0.0.1:5000
```

Upload a file or paste text, optionally paste a job description. Hover a skill to see its confidence and evidence.

**JSON API** (served by the web app)

```bash
curl -F resume=@resume.pdf -F "job_text=<job.txt" http://127.0.0.1:5000/api/extract
curl -H "Content-Type: application/json" -d '{"text": "...", "job_description": "..."}' http://127.0.0.1:5000/api/extract
```

**Python**

```python
from resume_skill_extractor import SkillExtractor, match_job

extractor = SkillExtractor()
result = extractor.extract_file("resume.pdf")
for skill in result.skills:
    print(skill.name, skill.category, skill.confidence, skill.evidence)

match = match_job(result, open("job.txt").read(), extractor)
print(match.score, match.missing_required)
```

## The taxonomy

[resume_skill_extractor/data/skills.json](resume_skill_extractor/data/skills.json) holds about 400 skills in 20 categories, from programming languages and cloud/DevOps to ML, data engineering, BI, security, design and soft skills. Each entry looks like this:

```json
{"name": "Go", "aliases": ["golang"], "strict": ["Go"]}
{"name": "React", "aliases": ["react.js", "reactjs", "react js"], "case_sensitive": ["React"]}
{"name": "Kubernetes", "aliases": ["k8s", "kubectl"]}
```

The canonical name is added as an alias automatically. On load the taxonomy is checked, and loading fails if two skills claim the same normalised alias. To add a skill, add an entry and run the tests.

## Evaluation

```bash
python -m resume_skill_extractor.evaluate          # uses eval/gold.jsonl
```

[eval/gold.jsonl](eval/gold.jsonl) is a small hand-labelled set of 14 snippets. They target the hard cases: aliases, hyphenation and plurals, symbol-heavy names, ambiguous words, name lines, all-caps lines and phrases broken across lines. The current result is **precision 0.967, recall 0.983, F1 0.975** (58 TP, 2 FP, 1 FN).

The set was written alongside the system, so treat it as a regression check rather than a benchmark. A fair evaluation needs a held-out set of real, annotated resumes.

The three remaining errors are kept in the set on purpose because they show real limitations:

| Case | Error | Why |
|---|---|---|
| "Software Engineer, **Oracle** – Redwood City" | false positive *Oracle Database* | an employer name next to a cue word looks like a skill |
| "trained a **sentiment classifier**" | false negative *Sentiment Analysis* | paraphrases are not matched; only listed aliases are |
| "presented results to senior **leadership**" | false positive *Leadership* | a soft-skill word used as an ordinary noun |

## Tests

```bash
python -m pytest
```

There are 38 tests. They cover tokenisation, the trie matcher, taxonomy validation, section detection, disambiguation, confidence scoring, job matching, the PDF/DOCX readers, the CLI, the Flask routes and the evaluation harness.

## Limitations and possible extensions

- **Dictionary-bound.** A skill is only found if it, or an alias, is in the taxonomy. Unknown skills are surfaced only when they appear in a skills list. Possible fixes are a trained NER model, such as a spaCy or BERT token classifier fine-tuned on annotated resumes, or embedding similarity to catch paraphrases like "sentiment classifier".
- **No negation or temporal reasoning.** "No experience with Java" still yields Java, and skills from 2010 count the same as current ones.
- **Layout.** Multi-column PDFs can come out with the columns interleaved, which can attach lines to the wrong section.
- **Implied skills** (Django ⇒ Python) are not inferred; the taxonomy could carry `implies` links for job matching.
- **No years of experience per skill** ("5 years of Python") are extracted yet.

## Project layout

```
resume_skill_extractor/
  readers.py      PDF / DOCX / TXT text extraction
  text.py         cleaning, tokenizer, normalisation
  sections.py     resume section segmentation
  taxonomy.py     taxonomy loading and validation
  matcher.py      token-trie longest-match phrase matcher
  extractor.py    pipeline, disambiguation, scoring, open-vocabulary skills
  jd_match.py     job-description matching
  cli.py          command line interface
  web.py          Flask app + JSON API (templates/index.html)
  evaluate.py     precision / recall / F1 on labelled data
  data/skills.json
samples/          example resume and job description
eval/gold.jsonl   labelled evaluation snippets
tests/            pytest suite
```
