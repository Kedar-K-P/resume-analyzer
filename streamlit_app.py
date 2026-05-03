"""
streamlit_app.py — Resume.AI on Streamlit Community Cloud

Wraps the existing pipeline_v2.py + ner_model_trainer.py pipeline
inside a Streamlit UI. Trains the NER model on first run (cached),
so cold starts take ~60 seconds; all subsequent runs are instant.
"""

import os
import sys
import json
import time
import tempfile
import streamlit as st

# ── Ensure project root is importable ─────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume.AI — ML Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark background */
  .stApp { background-color: #080b14; }
  section[data-testid="stSidebar"] { background-color: #0d1120; }

  /* Typography */
  html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; color: #e2e8f8; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #111827;
    border: 1px solid rgba(99,130,255,0.15);
    border-radius: 10px;
    padding: 14px !important;
  }
  [data-testid="metric-container"] label { color: #4a5568 !important; font-size: 11px !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #06b6d4 !important; font-size: 1.6rem !important; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #6366f1, #818cf8);
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 12px 28px;
    width: 100%;
    font-size: 13px;
  }
  .stButton > button:hover { opacity: 0.9; }

  /* Text areas */
  textarea {
    background: #111827 !important;
    border: 1px solid rgba(99,130,255,0.2) !important;
    border-radius: 8px !important;
    color: #e2e8f8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
  }

  /* Progress bars */
  .stProgress > div > div { background: #6366f1; border-radius: 4px; }

  /* Dividers */
  hr { border-color: rgba(99,130,255,0.12) !important; }

  /* Headers */
  h1, h2, h3 { color: #e2e8f8 !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #0d1120; border-bottom: 1px solid rgba(99,130,255,0.15); }
  .stTabs [data-baseweb="tab"] { color: #4a5568; font-size: 11px; letter-spacing: 2px; text-transform: uppercase; }
  .stTabs [aria-selected="true"] { color: #6366f1 !important; border-bottom: 2px solid #6366f1 !important; }

  /* File uploader */
  [data-testid="stFileUploader"] {
    background: #111827;
    border: 2px dashed rgba(99,130,255,0.25);
    border-radius: 10px;
    padding: 1rem;
  }

  /* Expanders */
  details { background: #111827; border: 1px solid rgba(99,130,255,0.12); border-radius: 8px; }
</style>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


# ── Model loading (cached — runs once per session) ─────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Train NER model if not present, then return initialized pipeline."""
    from pipeline_v2 import ResumeAnalyzerV2

    model_path = os.path.join(_HERE, "ner_model.pkl")
    dataset_path = os.path.join(_HERE, "data", "resume_dataset.json")

    # Step 1: Generate dataset if missing
    if not os.path.exists(dataset_path):
        st.toast("⚙️ Generating training dataset (first run)...")
        from dataset_generator import generate_dataset
        os.makedirs(os.path.join(_HERE, "data"), exist_ok=True)
        dataset = generate_dataset(2000)
        with open(dataset_path, "w") as f:
            json.dump(dataset, f)

    # Step 2: Train NER if model missing
    if not os.path.exists(model_path):
        st.toast("🧠 Training NER model (first run — ~60 seconds)...")
        from ner_model_trainer import train
        os.makedirs(os.path.join(_HERE, "evaluation"), exist_ok=True)
        train(
            dataset_path=dataset_path,
            model_save_path=model_path,
            eval_save_path=os.path.join(_HERE, "evaluation", "ner_eval_report.json"),
        )

    return ResumeAnalyzerV2(ner_model_path=model_path)


# ── Helpers ────────────────────────────────────────────────────────────────────
CHIP_COLORS = {
    "programming_languages": "#06b6d4",
    "ml_ai":                 "#818cf8",
    "cloud_devops":          "#10b981",
    "data_engineering":      "#f59e0b",
    "databases":             "#a78bfa",
    "data_visualization":    "#fb923c",
    "soft_skills":           "#fb7185",
}

def score_color(v):
    if v >= 75: return "#10b981"
    if v >= 55: return "#f59e0b"
    if v >= 40: return "#fb923c"
    return "#f43f5e"

def grade_color(g):
    if g in ("A+", "A"): return "#10b981"
    if g in ("B+", "B"): return "#f59e0b"
    if g == "C":         return "#fb923c"
    return "#f43f5e"

def chips_html(items, color):
    style = f"display:inline-block;background:{color}18;border:1px solid {color}44;color:{color};border-radius:20px;padding:2px 10px;margin:2px;font-size:11px;font-family:monospace"
    return " ".join(f'<span style="{style}">{s}</span>' for s in items)

def bar_html(label, value, color):
    return f"""
    <div style="margin:10px 0">
      <div style="display:flex;justify-content:space-between;font-size:11px;color:#718096;margin-bottom:5px;font-family:monospace">
        <span>{label}</span><span style="color:{color}">{value}%</span>
      </div>
      <div style="height:5px;background:rgba(255,255,255,0.06);border-radius:3px">
        <div style="height:100%;width:{value}%;background:{color};border-radius:3px;transition:width 1s"></div>
      </div>
    </div>"""

def suggestion_html(sg):
    colors = {"high": "#f43f5e", "medium": "#06b6d4", "low": "#4a5568"}
    c = colors.get(sg["priority"], "#4a5568")
    return f"""
    <div style="display:flex;align-items:flex-start;gap:10px;padding:10px 14px;border-radius:8px;
                background:{c}11;border-left:3px solid {c};margin-bottom:6px;font-size:12.5px">
      <span style="background:{c};color:{'white' if sg['priority']!='medium' else 'black'};
                   font-size:9px;padding:2px 7px;border-radius:4px;font-weight:700;
                   letter-spacing:1px;flex-shrink:0;margin-top:2px;font-family:monospace">
        {sg['priority'].upper()}
      </span>
      <span><strong style="color:#e2e8f8">{sg['category']}:</strong> {sg['action']}</span>
    </div>"""


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#0d1120;border-bottom:1px solid rgba(99,130,255,0.12);
            padding:14px 0 14px 0;margin-bottom:28px;display:flex;align-items:center;gap:12px">
  <div style="background:linear-gradient(135deg,#6366f1,#06b6d4);border-radius:8px;
              width:32px;height:32px;display:flex;align-items:center;justify-content:center;font-size:16px">🤖</div>
  <div>
    <span style="font-size:18px;font-weight:700;letter-spacing:0.5px">RESUME<span style="color:#06b6d4">.</span>AI</span>
    <span style="font-size:11px;color:#4a5568;margin-left:10px">v2.0 — Trained NER Model</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Load pipeline ──────────────────────────────────────────────────────────────
with st.spinner("🧠 Loading NER model (first run trains from scratch — ~60 seconds)..."):
    pipeline = load_pipeline()

ner_status = "✓ NER Active" if (pipeline.ner and pipeline.ner.loaded) else "⚠ Rule-based only"
st.caption(f"Model status: {ner_status}")


# ── Input section ──────────────────────────────────────────────────────────────
st.markdown("### 📋 Analyze Resume")

uploaded_file = st.file_uploader(
    "Upload resume (PDF, DOCX, TXT)",
    type=["pdf", "docx", "doc", "txt", "md"],
    help="Max 5 MB. PDF must be text-based (not scanned)."
)

col_r, col_j = st.columns(2)
with col_r:
    resume_text = st.text_area(
        "Or paste resume text",
        height=280,
        placeholder="John Doe\njohn@gmail.com | +91-9876543210\n\nSummary\nSenior Data Scientist...\n\nSkills\nPython, TensorFlow, AWS, Docker..."
    )
with col_j:
    jd_text = st.text_area(
        "Job Description (optional — for match score)",
        height=280,
        placeholder="We are looking for a Senior Data Scientist:\n• Strong Python and ML skills\n• Experience with NLP, BERT, transformers\n• AWS/GCP/Azure cloud experience\n• Docker and Kubernetes preferred"
    )

analyze_clicked = st.button("▶  Analyze Resume", use_container_width=True)


# ── Run analysis ───────────────────────────────────────────────────────────────
if analyze_clicked:
    result = None
    error = None

    with st.spinner("Analyzing with trained NER model..."):
        try:
            if uploaded_file is not None:
                file_bytes = uploaded_file.read()
                result = pipeline.analyze_file(file_bytes, uploaded_file.name, jd_text or "")
                if "error" in result:
                    error = result["error"]
                    result = None
            elif resume_text.strip():
                if len(resume_text.strip()) < 30:
                    error = "Please paste at least 30 characters of resume text."
                else:
                    result = pipeline.analyze(resume_text.strip(), jd_text or "")
            else:
                error = "Please upload a file or paste resume text."
        except Exception as e:
            error = str(e)

    if error:
        st.error(f"❌ {error}")

    if result:
        st.session_state["last_result"] = result
        st.success("✅ Analysis complete!")


# ── Display results ────────────────────────────────────────────────────────────
result = st.session_state.get("last_result")

if result:
    p = result.get("parsed", {})
    s = result.get("scores", {})
    contact   = p.get("contact", {})
    skills    = p.get("skills", {})
    impact    = p.get("impact", {})
    ner_ents  = p.get("ner_entities", {})
    sugs      = result.get("suggestions", [])
    jm        = result.get("job_match")

    st.divider()

    tab_results, tab_match, tab_ner, tab_model = st.tabs([
        "📊 Results", "🎯 Job Match", "🔍 NER Entities", "🧠 Model Info"
    ])

    # ── Results tab ─────────────────────────────────────────────────────────
    with tab_results:
        gc = grade_color(s.get("grade", "D"))

        # Score hero
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(99,102,241,0.12),rgba(6,182,212,0.08));
                    border:1px solid rgba(99,102,241,0.2);border-radius:14px;padding:2rem;
                    text-align:center;margin-bottom:1.5rem">
          <div style="font-size:11px;letter-spacing:3px;color:#4a5568;text-transform:uppercase;margin-bottom:10px;font-family:monospace">
            RESUME SCORE
          </div>
          <div style="font-size:4.5rem;font-weight:700;color:{gc};line-height:1;font-family:monospace">
            {s.get('overall', 0)}
          </div>
          <div style="font-size:1.5rem;font-weight:700;color:{gc};margin-top:6px;font-family:monospace">
            {s.get('grade', '—')}
          </div>
          <div style="font-size:10px;letter-spacing:2px;color:#4a5568;margin-top:8px;font-family:monospace">
            OUT OF 100 POINTS
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics row
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        mc1.metric("Skills",      p.get("total_skills", 0))
        mc2.metric("Categories",  len(skills))
        mc3.metric("Yrs Exp",     p.get("years_experience") or "—")
        mc4.metric("Words",       impact.get("word_count", 0))
        mc5.metric("Bullets",     impact.get("bullet_count", 0))
        mc6.metric("Metrics ✓",  "Yes" if impact.get("has_quantified_impact") else "No")

        # Contact
        if contact:
            st.markdown("**👤 Contact Detected**")
            contact_chips = " ".join(
                f'<span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);'
                f'color:#818cf8;border-radius:6px;padding:3px 12px;font-size:11px;font-family:monospace;margin:2px">'
                f'<span style="color:#4a5568">{k}:</span> {v}</span>'
                for k, v in contact.items() if v
            )
            st.markdown(contact_chips, unsafe_allow_html=True)
            st.markdown("")

        # Two-column layout: bars + suggestions
        left, right = st.columns(2)

        with left:
            st.markdown("**Score Breakdown**")
            bars_html = "".join(
                bar_html(label, s.get(key, 0), score_color(s.get(key, 0)))
                for label, key in [
                    ("Skills",       "skills_pct"),
                    ("Experience",   "experience_pct"),
                    ("Education",    "education_pct"),
                    ("Presentation", "presentation_pct"),
                    ("Impact",       "impact_pct"),
                ]
            )
            st.markdown(
                f'<div style="background:#111827;border:1px solid rgba(99,130,255,0.12);'
                f'border-radius:10px;padding:1.2rem">{bars_html}</div>',
                unsafe_allow_html=True
            )

        with right:
            st.markdown("**Improvements**")
            sugs_html = "".join(suggestion_html(sg) for sg in sugs[:8])
            st.markdown(
                f'<div style="background:#111827;border:1px solid rgba(99,130,255,0.12);'
                f'border-radius:10px;padding:1.2rem">{sugs_html}</div>',
                unsafe_allow_html=True
            )

        # Skills
        st.markdown("**Skills Detected**")
        if skills:
            for cat, cat_skills in skills.items():
                color = CHIP_COLORS.get(cat, "#718096")
                label = cat.replace("_", " ").title()
                st.markdown(
                    f'<div style="margin-bottom:10px">'
                    f'<div style="font-size:9px;color:#4a5568;letter-spacing:2px;text-transform:uppercase;margin-bottom:5px;font-family:monospace">{label}</div>'
                    f'{chips_html(cat_skills, color)}'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No skills detected. Check resume formatting.")

    # ── Job Match tab ────────────────────────────────────────────────────────
    with tab_match:
        if jm:
            mc = score_color(jm["match_percentage"])
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(16,185,129,0.1),rgba(6,182,212,0.06));
                        border:1px solid rgba(16,185,129,0.2);border-radius:14px;padding:2rem;
                        text-align:center;margin-bottom:1.5rem">
              <div style="font-size:11px;letter-spacing:3px;color:#4a5568;text-transform:uppercase;margin-bottom:10px;font-family:monospace">JOB MATCH SCORE</div>
              <div style="font-size:4rem;font-weight:700;color:{mc};font-family:monospace">{jm['match_percentage']}%</div>
              <div style="font-size:13px;color:#718096;margin-top:12px;max-width:480px;margin-left:auto;margin-right:auto">{jm['recommendation']}</div>
              <div style="font-size:11px;color:#4a5568;margin-top:10px;font-family:monospace">{jm.get('jd_skills_found', 0)} required skills detected in JD</div>
            </div>
            """, unsafe_allow_html=True)

            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown("**✅ Matched Skills**")
                if jm["matched_skills"]:
                    st.markdown(chips_html(jm["matched_skills"], "#10b981"), unsafe_allow_html=True)
                else:
                    st.caption("No direct skill matches found.")
            with mc2:
                st.markdown("**❌ Missing Skills — Add These**")
                if jm["missing_skills"]:
                    st.markdown(chips_html(jm["missing_skills"], "#f43f5e"), unsafe_allow_html=True)
                else:
                    st.success("No major skill gaps!")
        else:
            st.info("Paste a job description in the input area and re-run analysis to see your match score.")

    # ── NER Entities tab ─────────────────────────────────────────────────────
    with tab_ner:
        NER_COLORS = {
            "PERSON": "#06b6d4", "EMAIL": "#10b981", "PHONE": "#f59e0b",
            "URL": "#818cf8", "COMPANY": "#f59e0b", "JOB_TITLE": "#fb7185",
            "UNIVERSITY": "#10b981", "DEGREE": "#a78bfa", "CERTIFICATION": "#fb923c",
            "SKILL": "#718096",
        }
        label_map = {"companies": "COMPANY", "job_titles": "JOB_TITLE"}

        if ner_ents:
            for ent_type, values in ner_ents.items():
                if not values:
                    continue
                display = label_map.get(ent_type, ent_type.upper())
                color   = NER_COLORS.get(display, "#718096")
                st.markdown(
                    f'<div style="margin-bottom:14px">'
                    f'<div style="font-size:9px;color:#4a5568;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;font-family:monospace">{display.replace("_"," ")}</div>'
                    f'{chips_html(values, color)}'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div style="background:#111827;border:1px solid rgba(99,130,255,0.12);border-radius:10px;padding:1.5rem">'
                '<div style="color:#4a5568;font-size:13px">NER model ran — entities are merged into the parsed output above.<br><br>'
                'If the NER model is not loaded, rule-based contact extraction is used instead.</div>'
                '</div>', unsafe_allow_html=True
            )
            if contact:
                st.markdown("**Rule-based Contact Extraction**")
                for k, v in contact.items():
                    if v:
                        st.markdown(f'`{k}:` {v}')

    # ── Model Info tab ───────────────────────────────────────────────────────
    with tab_model:
        st.markdown("**Architecture**")
        arch = {
            "Type":      "Token-classification NER — Logistic Regression + 200+ features",
            "Training":  "2,000 synthetic resume samples · 1,800 train / 200 test",
            "Features":  "Shape, context window, lexicon lookup, n-gram prefix/suffix",
            "Labels":    "PERSON, EMAIL, PHONE, URL, SKILL, COMPANY, JOB_TITLE, UNIVERSITY, DEGREE, CERT",
            "Overall F1": "91.5%",
        }
        for k, v in arch.items():
            st.markdown(f"**`{k}`** &nbsp; {v}")

        st.divider()
        st.markdown("**Per-Entity F1 (held-out test set)**")

        entity_scores = [
            ("B-PERSON", 100), ("B-EMAIL", 100), ("B-PHONE", 100), ("B-URL", 100),
            ("B-DEGREE", 80), ("B-CERTIFICATION", 78.8), ("B-SKILL", 73.5),
            ("B-UNIVERSITY", 58.8), ("B-COMPANY", 56.0), ("B-JOB_TITLE", 10.1),
        ]

        # Try loading live eval report
        eval_path = os.path.join(_HERE, "evaluation", "ner_eval_report.json")
        if os.path.exists(eval_path):
            try:
                with open(eval_path) as f:
                    eval_data = json.load(f)
                live = eval_data.get("per_entity", {})
                if live:
                    entity_scores = [(k, round(v * 100, 1)) for k, v in live.items()]
            except Exception:
                pass

        bars_html = "".join(bar_html(l, v, score_color(v)) for l, v in entity_scores)
        st.markdown(
            f'<div style="background:#111827;border:1px solid rgba(99,130,255,0.12);'
            f'border-radius:10px;padding:1.2rem">{bars_html}</div>',
            unsafe_allow_html=True
        )

        st.divider()
        st.markdown("**Upgrade Path**")
        upgrades = [
            "Fine-tune `bert-base-uncased` on resume corpus → F1: 91.5% → **95%+**",
            "Add real Kaggle Resume Dataset (2,400 samples) for better generalization",
            "Add `sentence-transformers` for semantic (not keyword) job matching",
            "Add Tesseract OCR for scanned PDF resume support",
            "Add PostgreSQL to persist analysis history across sessions",
        ]
        for u in upgrades:
            st.markdown(f"→ {u}")

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:4rem;color:#4a5568">
      <div style="font-size:3rem;margin-bottom:16px">📄</div>
      <div style="font-size:14px">Upload a resume or paste text above, then click <strong style="color:#6366f1">Analyze Resume</strong></div>
      <div style="font-size:12px;margin-top:8px">Supports PDF · DOCX · TXT · Markdown</div>
    </div>
    """, unsafe_allow_html=True)
