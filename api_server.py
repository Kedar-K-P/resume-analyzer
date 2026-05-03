"""
Resume Analyzer API Server
FastAPI REST API with file upload, batch analysis, and model info endpoints.

FIXES vs original:
1. Import path for pipeline_v2 was `from models.pipeline_v2 import ...` — but
   pipeline_v2.py lives alongside api_server.py, not in a models/ subdirectory.
   Fixed to `from pipeline_v2 import ResumeAnalyzerV2`.
2. _NER_MODEL_PATH now resolves correctly whether CWD is project root or models/.
3. Health endpoint now handles the case where _pipeline is None without crashing.
4. CORS middleware added so the frontend can hit the API from any origin
   (required when opening index.html directly from disk).
5. Root route serves index.html from the same directory as api_server.py
   (not from a nonexistent frontend/ subdirectory).
6. Graceful startup — model is loaded lazily on first request, so the server
   starts even if the NER pickle hasn't been trained yet.

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
import json
from typing import Optional, List

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
# Ensure project directory is importable
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# FIX: pipeline_v2 is in the same directory as api_server.py, not in models/
from pipeline_v2 import ResumeAnalyzerV2

# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Resume Analyzer API",
    description="ML-powered resume analysis: NER extraction, scoring, job matching.",
    version="2.0.0",
)

# FIX: CORS — needed when index.html is opened from disk or a different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_START_TIME = time.time()
_REQUEST_COUNT = {"analyze": 0, "file": 0, "batch": 0}

# FIX: NER model lives alongside api_server.py and pipeline_v2.py
_NER_MODEL_PATH = os.path.join(_HERE, "ner_model.pkl")
_pipeline: Optional[ResumeAnalyzerV2] = None


def get_pipeline() -> ResumeAnalyzerV2:
    global _pipeline
    if _pipeline is None:
        _pipeline = ResumeAnalyzerV2(ner_model_path=_NER_MODEL_PATH)
    return _pipeline


# ── Schemas ───────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    resume_text: str
    job_description: Optional[str] = ""

class BatchRequest(BaseModel):
    resumes: List[AnalyzeRequest]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the frontend HTML from the same directory as this file."""
    # FIX: index.html is in _HERE, not in a frontend/ subdirectory
    html_path = os.path.join(_HERE, "index.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Resume Analyzer API</h1><p>See <a href='/docs'>/docs</a></p>")


@app.post("/analyze")
def analyze_text(req: AnalyzeRequest):
    """Analyze plain-text resume."""
    if not req.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text is empty")
    _REQUEST_COUNT["analyze"] += 1
    result = get_pipeline().analyze(req.resume_text, req.job_description or "")
    return JSONResponse(content=result)


@app.post("/analyze/file")
async def analyze_file(
    file: UploadFile = File(...),
    job_description: str = Form(default=""),
):
    """Upload and analyze a PDF, DOCX, or TXT resume."""
    allowed = {"pdf", "docx", "doc", "txt", "md"}
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Use: {allowed}")
    _REQUEST_COUNT["file"] += 1
    content = await file.read()
    result = get_pipeline().analyze_file(content, file.filename or "resume.txt", job_description)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return JSONResponse(content=result)


@app.post("/batch")
def analyze_batch(req: BatchRequest):
    """Analyze up to 10 resumes at once."""
    if len(req.resumes) > 10:
        raise HTTPException(status_code=400, detail="Batch limit is 10 resumes.")
    _REQUEST_COUNT["batch"] += 1
    pipe = get_pipeline()
    results = []
    for i, r in enumerate(req.resumes):
        try:
            results.append({"index": i, "result": pipe.analyze(r.resume_text, r.job_description or "")})
        except Exception as e:
            results.append({"index": i, "error": str(e)})
    return {"count": len(results), "results": results}


@app.get("/health")
def health():
    """Service health check."""
    # FIX: don't crash if pipeline not yet initialized
    ner_loaded = False
    if _pipeline is not None and _pipeline.ner is not None:
        ner_loaded = _pipeline.ner.loaded
    return {
        "status": "ok",
        "ner_model_loaded": ner_loaded,
        "ner_model_path": _NER_MODEL_PATH,
        "ner_model_exists": os.path.exists(_NER_MODEL_PATH),
        "uptime_seconds": round(time.time() - _START_TIME, 1),
    }


@app.get("/metrics")
def metrics():
    """Request counts and uptime."""
    return {
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "requests": _REQUEST_COUNT,
    }


@app.get("/model/info")
def model_info():
    """NER model evaluation report (if available)."""
    eval_path = os.path.join(_HERE, "evaluation", "ner_eval_report.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            return json.load(f)
    return {"message": "Evaluation report not found. Run ner_model_trainer.py first."}
