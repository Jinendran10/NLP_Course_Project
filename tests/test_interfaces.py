import io
import json

import pytest

from resume_skill_extractor import read_file
from resume_skill_extractor.cli import main
from resume_skill_extractor.evaluate import evaluate
from resume_skill_extractor.web import create_app


# -- readers ------------------------------------------------------------------


def test_reads_docx_paragraphs_and_tables(tmp_path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Skills")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Python"
    table.cell(0, 1).text = "Docker"
    path = tmp_path / "cv.docx"
    document.save(path)
    text = read_file(path)
    assert "Skills" in text and "Python | Docker" in text


def test_reads_pdf(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Skills: Python, Kubernetes")
    path = tmp_path / "cv.pdf"
    document.save(path)
    document.close()
    assert "Kubernetes" in read_file(path)


def test_rejects_unsupported_and_empty_files(tmp_path):
    (tmp_path / "cv.odt").write_bytes(b"x")
    (tmp_path / "empty.txt").write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported file type"):
        read_file(tmp_path / "cv.odt")
    with pytest.raises(ValueError, match="no text"):
        read_file(tmp_path / "empty.txt")


# -- CLI ----------------------------------------------------------------------


def test_cli_json_with_job(samples, capsys):
    code = main([str(samples / "sample_resume.txt"), "--jd", str(samples / "sample_jd.txt"), "--json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert "Python" in [s["name"] for s in data["skills"]]
    assert data["job_match"]["missing_required"] == ["BigQuery", "Infrastructure as Code"]


def test_cli_text_report_and_ranking(samples, tmp_path, capsys):
    other = tmp_path / "other.txt"
    other.write_text("Skills\nJava, Kubernetes", encoding="utf-8")
    code = main([str(samples / "sample_resume.txt"), str(other), "--jd", str(samples / "sample_jd.txt")])
    out = capsys.readouterr().out
    assert code == 0
    assert "Programming Languages" in out and "Job match" in out
    ranking = out[out.index("Ranking against"):]
    assert ranking.index("sample_resume.txt") < ranking.index("other.txt")


def test_cli_reports_unreadable_files(tmp_path, capsys):
    assert main([str(tmp_path / "missing.pdf")]) == 1
    assert "missing.pdf" in capsys.readouterr().err


# -- web ----------------------------------------------------------------------


@pytest.fixture
def client(extractor):
    return create_app(extractor).test_client()


def test_form_with_pasted_text_and_job(client):
    response = client.post("/", data={"resume_text": "Skills\nPython, Docker", "job_text": "Requirements:\nPython, Kubernetes"})
    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Python" in body and "Kubernetes" in body and "Job match" in body


def test_form_with_file_upload(client):
    data = {"resume": (io.BytesIO(b"Skills\nRust, Terraform"), "cv.txt")}
    response = client.post("/", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert "Terraform" in response.get_data(as_text=True)


def test_empty_form_shows_error(client):
    response = client.post("/", data={"resume_text": ""})
    assert response.status_code == 400
    assert "Upload a resume" in response.get_data(as_text=True)


def test_json_api(client):
    response = client.post("/api/extract", json={"text": "Skills\nPython", "job_description": "Requirements:\nPython"})
    data = response.get_json()
    assert response.status_code == 200
    assert data["skills"][0]["name"] == "Python"
    assert data["job_match"]["score"] > 80


def test_json_api_rejects_bad_upload(client):
    data = {"resume": (io.BytesIO(b"x"), "cv.doc")}
    response = client.post("/api/extract", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert ".doc" in response.get_json()["error"]


# -- evaluation ---------------------------------------------------------------


def test_evaluation_runs_on_gold_set(extractor):
    report = evaluate(extractor=extractor)
    assert report["precision"] > 0.9 and report["recall"] > 0.9
