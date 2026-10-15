"""Vercel serverless entry point for the Flask application."""

from resume_skill_extractor.web import create_app

app = create_app()
