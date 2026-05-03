# Resume.AI — ML-Powered Resume Analyzer

A production-ready resume analysis system combining a **custom-trained NER model** with rule-based extraction, scoring, job-description matching, and a modern dark-theme frontend.

---

## Architecture

```
.
├── dataset_generator.py     # Generates 2,000 synthetic labeled resume samples
├── ner_model_trainer.py     # Trains token-classification NER (Logistic Regression)
├── pipeline_v2.py           # Full analysis pipeline (NER + rules + scoring + matching)
├── document_parser.py       # Pure-Python PDF / DOCX / RTF / TXT parser (no deps)
├── api_server.py            # FastAPI REST API — serves the frontend and JSON endpoints
├── index.html               # Single-file frontend (vanilla JS, no build step)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Multi-stage build: generate → train → serve
└── README.md
```

Generated at runtime:
```
data/
└── resume_dataset.json      # 2,000 labeled samples (created by dataset_generator.py)
evaluation/
└── ner_eval_report.json     # Per-entity F1 scores (created by ner_model_trainer.py)
ner_model.pkl                # Trained NER model pickle (created by ner_model_trainer.py)
```

---

## Quick Start (local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the training dataset

```bash
python dataset_generator.py
# → writes data/resume_dataset.json  (2,000 samples)
```

### 3. Train the NER model

```bash
python ner_model_trainer.py
# → writes ner_model.pkl
# → writes evaluation/ner_eval_report.json
```

Training takes ~30–60 seconds on a modern laptop.

### 4. Start the API server

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open the frontend

Navigate to **http://localhost:8000** — the server serves `index.html` from the same directory.

The frontend also works **fully offline** (client-side JS analysis) if the API is unavailable.

---

## Docker

```bash
docker build -t resume-ai .
docker run -p 8000:8000 resume-ai
```

The Dockerfile automatically runs `dataset_generator.py` and `ner_model_trainer.py` during the image build, so the container starts with a trained model ready.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Serves `index.html` |
| `POST` | `/analyze` | Analyze plain-text resume (JSON body) |
| `POST` | `/analyze/file` | Upload PDF / DOCX / TXT file (multipart) |
| `POST` | `/batch` | Analyze up to 10 resumes at once |
| `GET`  | `/health` | Service health + NER model status |
| `GET`  | `/metrics` | Request counts + uptime |
| `GET`  | `/model/info` | Live NER evaluation report |
| `GET`  | `/docs` | Interactive Swagger UI |

### Example — analyze text

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Jane Doe\njane@email.com\nPython, TensorFlow, AWS\n5 years experience",
    "job_description": "Looking for Python developer with ML experience"
  }'
```

### Example — upload file

```bash
curl -X POST http://localhost:8000/analyze/file \
  -F "file=@/path/to/resume.pdf" \
  -F "job_description=Senior ML Engineer with AWS experience"
```

### Response shape

```json
{
  "parsed": {
    "contact":   { "name": "...", "email": "...", "phone": "...", "linkedin": "...", "github": "..." },
    "skills":    { "programming_languages": [...], "ml_ai": [...], "cloud_devops": [...] },
    "education": [ { "degree": "MTECH" } ],
    "impact":    { "has_quantified_impact": true, "word_count": 420, "bullet_count": 9 },
    "years_experience": 5,
    "ner_entities": { "PERSON": [...], "COMPANY": [...], "JOB_TITLE": [...] },
    "total_skills": 18,
    "skill_categories": ["programming_languages", "ml_ai", "cloud_devops", "databases"]
  },
  "scores": {
    "overall": 76.5,
    "grade": "A",
    "skills_pct": 85,
    "experience_pct": 72,
    "education_pct": 85,
    "presentation_pct": 65,
    "impact_pct": 70
  },
  "suggestions": [
    { "priority": "high",   "category": "Contact", "action": "Add your LinkedIn profile URL" },
    { "priority": "medium", "category": "Skills",  "action": "Add cloud experience (AWS/GCP/Azure/Docker)" }
  ],
  "job_match": {
    "match_percentage": 82,
    "matched_skills":   ["python", "tensorflow", "aws"],
    "missing_skills":   ["kubernetes", "mlflow"],
    "recommendation":   "Excellent match! Apply with confidence.",
    "jd_skills_found":  5
  }
}
```

---

## NER Model

### Labels

| Label | Examples |
|-------|---------|
| `PERSON` | Priya Sharma, John Doe |
| `EMAIL` | priya@gmail.com |
| `PHONE` | +91-9876543210 |
| `URL` | linkedin.com/in/priya, github.com/user |
| `SKILL` | Python, TensorFlow, Docker |
| `COMPANY` | Google, Flipkart, Microsoft |
| `JOB_TITLE` | Senior Data Scientist, ML Engineer |
| `UNIVERSITY` | IIT Bombay, Stanford University |
| `DEGREE` | M.Tech, B.S., Ph.D |
| `CERTIFICATION` | AWS Certified Machine Learning Specialty |

### Performance (held-out test set)

| Entity | F1 |
|--------|----|
| EMAIL | 100% |
| PERSON | 100% |
| PHONE | 100% |
| URL | 100% |
| DEGREE | 80.0% |
| CERTIFICATION | 78.8% |
| SKILL | 73.5% |
| UNIVERSITY | 58.8% |
| COMPANY | 56.0% |
| JOB_TITLE | 10.1% |
| **Weighted avg** | **91.5%** |

JOB_TITLE F1 is low because job titles overlap heavily with common English words — a fine-tuned BERT model would solve this.

---

## Bug Fixes (vs original code)

### `ner_model_trainer.py`
- **`LogisticRegression` `multi_class='multinomial'`** — removed deprecated kwarg that raises `TypeError` in scikit-learn ≥ 1.5. Solver `lbfgs` handles multiclass natively.
- **Solver `saga` → `lbfgs`** — `saga` doesn't support `multinomial` on small datasets and was causing convergence warnings.
- **`n_jobs=-1` with `lbfgs`** — `lbfgs` ignores `n_jobs`; changed to `n_jobs=1` to avoid misleading parallelism.
- **Feature vocabulary from first sample only** — `collect_feature_names()` now scans ALL training samples, preventing silent zero-fill for prefix/suffix n-grams seen only in later samples.
- **Orphan `I-` tags** — `extract_entities()` now treats an `I-X` tag with no preceding `B-X` as a new entity start instead of silently dropping it.
- **`os.makedirs` missing `abspath`** — `save()` now wraps path in `abspath` so relative paths don't fail on nested invocations.

### `pipeline_v2.py`
- **Pickle key validation** — `NERExtractor._load()` validates all required keys exist before unpacking, instead of crashing with an opaque `KeyError`.
- **`feature_names is None` guard** — `NERExtractor.extract()` now checks `feature_names` before inference.
- **Orphan `I-` tag handling** — same fix as trainer.
- **`analyze_file()` path injection** — `_ROOT` is now prepended with `insert(0, ...)` so the project directory always wins over site-packages.

### `api_server.py`
- **Wrong import path** — `from models.pipeline_v2 import ...` → `from pipeline_v2 import ...` (both files are in the same directory).
- **CORS missing** — added `CORSMiddleware` so the frontend works when opened from disk or a different origin.
- **`/health` crash** — fixed crash when `_pipeline` is `None` (before first request).
- **`index.html` path** — was looking in a nonexistent `frontend/` subdirectory; fixed to look in the same directory as `api_server.py`.

### `document_parser.py`
- **PDF duplicate strings** — stream-level fallback was re-collecting strings already captured by the BT/ET parser; fixed by tracking seen character positions.
- **PDF fallback junk** — printable-ASCII fallback now skips PDF operator tokens (`obj`, `endobj`, `stream`, `Tj`, etc.).
- **DOCX `<w:t>` secondary path** — added explicit text-run collection for documents where paragraph tags are present but paragraph-aware stripping yields empty output.
- **RTF markup passthrough** — `_parse_rtf()` now strips RTF control words instead of returning raw markup to the scorer.
- **`base64` deferred import** — `base64` is now imported at module level.

### `dataset_generator.py`
- **CWD-relative output path** — all output paths now resolve relative to `__file__`, not `os.getcwd()`, so the script works from any working directory.
- **Missing `__main__` guard** — generation no longer runs on `import`.

### `index.html`
- **HTML injection** — all user/API data rendered into the DOM now passes through `escHtml()`.
- **`switchTab()` hardcoded index** — was calling `document.querySelectorAll('.tab')[1].click()` to switch to Results; replaced with `switchTab('results', resultsTab)`.
- **File upload offline error** — now shows a clear message instead of silently sending an empty string to the client-side analyzer.
- **API auto-detection** — polls `/health` every 30 seconds and updates the status badge live.
- **`${renderEntityBars()}`** — was a template literal in a static HTML string (not a JS template); moved to a proper `initEntityBars()` call on page load.

---

## Upgrade Path

| Upgrade | Expected improvement |
|---------|---------------------|
| Fine-tune `bert-base-uncased` on resume corpus | NER F1: 91.5% → 95%+ |
| Add real Kaggle Resume Dataset (2,400 samples) | Better generalization on real resumes |
| `sentence-transformers` for semantic job matching | Match quality: keyword → semantic |
| Tesseract OCR integration | Support scanned PDF resumes |
| PostgreSQL for analysis history | Persistent multi-session storage |
| Redis caching for repeated resumes | Sub-100ms repeated analysis |

---

## Development

```bash
# Run tests (if added)
pytest tests/ -v

# Lint
ruff check .

# Type-check
mypy api_server.py pipeline_v2.py ner_model_trainer.py
```

---

## License

MIT
