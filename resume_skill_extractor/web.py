"""Small Flask front end and JSON API.

Run with ``python -m resume_skill_extractor.web`` and open http://127.0.0.1:5000.
"""

from __future__ import annotations

import argparse

from flask import Flask, jsonify, render_template, request

from .extractor import SkillExtractor
from .jd_match import match_job
from .readers import SUPPORTED_EXTENSIONS, read_bytes

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def create_app(extractor: SkillExtractor | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    extractor = extractor or SkillExtractor()
    accept = ",".join(SUPPORTED_EXTENSIONS)

    def analyse(resume_text: str, job_text: str):
        if not resume_text.strip():
            raise ValueError("Upload a resume file or paste its text.")
        result = extractor.extract(resume_text)
        match = match_job(result, job_text, extractor) if job_text.strip() else None
        return result, match

    def uploaded_resume_text() -> str:
        upload = request.files.get("resume")
        if upload and upload.filename:
            return read_bytes(upload.read(), upload.filename)
        return request.form.get("resume_text", "")

    @app.get("/")
    def index():
        return render_template("index.html", form={}, accept=accept)

    @app.post("/")
    def analyse_form():
        form = {"resume_text": request.form.get("resume_text", ""), "job_text": request.form.get("job_text", "")}
        try:
            result, match = analyse(uploaded_resume_text(), form["job_text"])
        except ValueError as exc:
            return render_template("index.html", form=form, accept=accept, error=str(exc)), 400
        return render_template("index.html", form=form, accept=accept, result=result, match=match)

    @app.post("/api/extract")
    def api_extract():
        """JSON body {"text": ..., "job_description": ...} or a multipart upload
        with a "resume" file (or "resume_text") and optional "job_text"."""
        try:
            if request.is_json:
                payload = request.get_json(silent=True) or {}
                resume_text, job_text = payload.get("text") or "", payload.get("job_description") or ""
            else:
                resume_text, job_text = uploaded_resume_text(), request.form.get("job_text", "")
            result, match = analyse(resume_text, job_text)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        body = result.to_dict()
        if match is not None:
            body["job_match"] = match.to_dict()
        return jsonify(body)

    @app.errorhandler(413)
    def too_large(_):
        message = f"File too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)."
        if request.path.startswith("/api/"):
            return jsonify(error=message), 413
        return render_template("index.html", form={}, accept=accept, error=message), 413

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the resume skill extractor web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    create_app().run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
