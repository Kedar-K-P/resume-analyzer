"""
streamlit_app.py — Resume.AI  (Streamlit Cloud Edition)
========================================================
ZERO external ML dependencies. Pure Python + Streamlit only.
No sklearn, no pickle, no model training, no subprocess calls.
All analysis runs instantly with regex + taxonomy matching.
Works on Streamlit Community Cloud free tier (512 MB RAM).
"""

import re
import io
import zipfile
import streamlit as st

st.set_page_config(
    page_title="Resume.AI — Smart Resume Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@700;800&display=swap');
.stApp { background-color: #08090f !important; }
html, body { font-family: 'JetBrains Mono', monospace !important; }
h1,h2,h3 { color: #e2e8f8 !important; }
p, li { color: #94a3b8; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1100px; }
.stTextArea textarea {
    background: #111827 !important; border: 1px solid rgba(74,244,232,0.15) !important;
    border-radius: 10px !important; color: #e2e8f8 !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
}
.stTextArea textarea:focus { border-color: rgba(74,244,232,0.45) !important; box-shadow: 0 0 0 3px rgba(74,244,232,0.07) !important; }
[data-testid="stFileUploader"] { background: #111827; border: 1.5px dashed rgba(74,244,232,0.25); border-radius: 12px; padding: 0.5rem; }
.stButton > button {
    background: #4af4e8 !important; color: #000 !important; border: none !important;
    border-radius: 10px !important; font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important; letter-spacing: 2px !important;
    text-transform: uppercase !important; width: 100% !important; padding: 14px !important; font-size: 13px !important;
}
.stButton > button:hover { background: #72f8f3 !important; box-shadow: 0 6px 20px rgba(74,244,232,0.22) !important; }
[data-testid="metric-container"] { background: #111827 !important; border: 1px solid rgba(74,244,232,0.12) !important; border-radius: 10px !important; padding: 1rem !important; }
[data-testid="stMetricLabel"] { color: #4a5568 !important; font-size: 11px !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricValue"] { color: #4af4e8 !important; font-size: 1.4rem !important; font-weight: 700 !important; }
.stTabs [data-baseweb="tab-list"] { background: #0d1220 !important; border-bottom: 1px solid rgba(74,244,232,0.1) !important; }
.stTabs [data-baseweb="tab"] { color: #4a5568 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; letter-spacing: 2px !important; text-transform: uppercase !important; }
.stTabs [aria-selected="true"] { color: #4af4e8 !important; border-bottom: 2px solid #4af4e8 !important; background: transparent !important; }
details { background: #111827 !important; border: 1px solid rgba(74,244,232,0.1) !important; border-radius: 10px !important; }
summary { color: #94a3b8 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }
hr { border-color: rgba(74,244,232,0.1) !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: rgba(74,244,232,0.15); border-radius: 3px; }
.stAlert { border-radius: 8px !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── SKILL TAXONOMY ─────────────────────────────────────────────────────────────
SKILL_TAXONOMY = {
    "Programming Languages": [
        "python","java","javascript","typescript","c++","c#","go","rust","kotlin","swift",
        "r","scala","ruby","php","sql","bash","html","css","matlab","perl","dart"
    ],
    "ML / AI": [
        "machine learning","deep learning","nlp","natural language processing","bert","gpt",
        "transformers","huggingface","tensorflow","pytorch","keras","scikit-learn","sklearn",
        "xgboost","lightgbm","catboost","reinforcement learning","llm","openai","langchain",
        "cnn","lstm","rnn","computer vision","yolo","generative ai","rag","stable diffusion"
    ],
    "Data Engineering": [
        "apache spark","kafka","airflow","hadoop","dbt","etl","elt","data pipeline",
        "snowflake","databricks","pyspark","pandas","numpy","dask","hive","flink","data warehouse"
    ],
    "Cloud & DevOps": [
        "aws","azure","gcp","google cloud","docker","kubernetes","terraform","ansible",
        "ci/cd","jenkins","github actions","linux","fastapi","flask","django","spring boot",
        "mlflow","mlops","microservices","rest api","grpc","prometheus","grafana"
    ],
    "Databases": [
        "postgresql","mysql","mongodb","redis","cassandra","elasticsearch","bigquery",
        "redshift","dynamodb","firebase","neo4j","sqlite","oracle","pinecone"
    ],
    "Visualization": [
        "tableau","power bi","matplotlib","seaborn","plotly","d3.js","looker",
        "grafana","bokeh","dash","looker studio","superset"
    ],
    "Methods": [
        "agile","scrum","a/b testing","statistics","hypothesis testing",
        "project management","team leadership","system design","data structures","algorithms"
    ],
}

IMPACT_VERBS = [
    "built","developed","designed","created","architected","engineered","led","managed",
    "improved","optimized","reduced","increased","deployed","launched","delivered",
    "implemented","automated","scaled","shipped","accelerated","analyzed","researched",
    "mentored","collaborated","established","streamlined","transformed","drove"
]

DEGREE_PAT  = re.compile(r'\b(b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?|b\.?s\.?|m\.?s\.?|ph\.?d|mba|bachelor|master|doctorate|b\.?sc|m\.?sc)\b', re.I)
SECTION_PAT = re.compile(r'\b(summary|profile|objective|about|overview)\b', re.I)
METRIC_PAT  = re.compile(r'\d+\s*%|\$\s*\d+|\d+\s*x\b|\d+\s*(million|thousand|\bk\b)', re.I)
BULLET_PAT  = re.compile(r'^[\s]*[•\-\*]\s*.+', re.M)

# ── PARSERS ────────────────────────────────────────────────────────────────────
def parse_document(file_bytes, filename):
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "txt"
    if ext == "pdf":
        return _parse_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        return _parse_docx(file_bytes)
    elif ext == "rtf":
        text = file_bytes.decode("utf-8", errors="replace")
        text = re.sub(r"\\[a-zA-Z]+\d*\s?", " ", text)
        return re.sub(r"\s+", " ", text.replace("{","").replace("}","")).strip()
    else:
        return file_bytes.decode("utf-8", errors="replace")

def _parse_pdf(data):
    try:
        content = data.decode("latin-1", errors="replace")
        parts = []
        for block in re.finditer(r"BT\s*(.*?)\s*ET", content, re.DOTALL):
            for m in re.finditer(r"\(([^)\\]*(?:\\.[^)\\]*)*)\)", block.group(1)):
                s = m.group(1).replace("\\n","\n").replace("\\r","").replace("\\t"," ").replace("\\\\","\\")
                if s.strip() and len(s) > 1:
                    parts.append(s)
        text = re.sub(r"\s+", " ", " ".join(parts))
        text = re.sub(r"[^\x20-\x7E\n]", " ", text)
        if len(text.strip()) < 60:
            candidates = re.findall(r"[A-Za-z0-9@.\-_+:,()\s]{6,}", content)
            junk = {"obj","endobj","stream","endstream","xref","trailer","BT","ET","Tf","Td"}
            text = "\n".join(c.strip() for c in candidates if c.strip() not in junk and len(c.strip()) > 4)
        return text.strip()
    except Exception:
        return ""

def _parse_docx(data):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="replace")
        out = re.sub(r"<w:p[ >]", "\n", xml)
        out = re.sub(r"<w:br[^/]*/?>", "\n", out)
        out = re.sub(r"<w:tab/?>", "\t", out)
        out = re.sub(r"<[^>]+>", "", out)
        for e,c in [("&amp;","&"),("&lt;","<"),("&gt;",">"),("&quot;",'"'),("&apos;","'")]:
            out = out.replace(e, c)
        lines = [l.strip() for l in out.split("\n") if l.strip()]
        result = "\n".join(lines)
        if len(result.strip()) < 50:
            wt = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml, re.DOTALL)
            result = " ".join(wt)
        return result.strip()
    except Exception:
        return ""

# ── ANALYSIS ───────────────────────────────────────────────────────────────────
def _esc_re(s):
    return re.escape(s)

def extract_contact(text):
    c = {}
    m = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    if m: c["email"] = m.group()
    m = re.search(r'\+?[\d][\d\s\-\(\)\.]{8,14}\d', text)
    if m: c["phone"] = m.group().strip()
    m = re.search(r'linkedin\.com/in/[^\s\n,)\]]+', text, re.I)
    if m: c["linkedin"] = m.group()
    m = re.search(r'github\.com/[^\s\n,)\]]+', text, re.I)
    if m: c["github"] = m.group()
    for line in text.split('\n')[:8]:
        l = line.strip()
        if 4 < len(l) < 60 and re.match(r'^[A-Za-z][A-Za-z\s\-\.]+$', l):
            ws = l.split()
            if 2 <= len(ws) <= 5 and all(w[0].isupper() for w in ws if w.isalpha()):
                c["name"] = l; break
    return c

def extract_skills(text):
    tl = text.lower()
    out, seen = {}, set()
    for cat, skills in SKILL_TAXONOMY.items():
        found = []
        for s in sorted(skills, key=len, reverse=True):
            if re.search(r'\b' + _esc_re(s) + r'\b', tl, re.I) and s not in seen:
                found.append(s); seen.add(s)
        if found:
            out[cat] = found
    return out

def score_resume(text, contact, skills):
    total_sk = sum(len(v) for v in skills.values())
    cats     = len(skills)
    has_adv  = bool(skills.get("ML / AI") or skills.get("Cloud & DevOps"))
    wc       = len(text.split())
    bullets  = len(BULLET_PAT.findall(text))
    has_met  = bool(METRIC_PAT.search(text))
    has_sum  = bool(SECTION_PAT.search(text))
    has_deg  = bool(DEGREE_PAT.search(text))
    m = re.search(r'(\d+)\+?\s*(?:years?|yrs?)[\s\w]{0,12}(?:experience|exp)', text, re.I)
    yrs = int(m.group(1)) if m else 0
    verbs = [v for v in IMPACT_VERBS if re.search(r'\b' + v + r'\b', text, re.I)]

    sk_sc  = min(total_sk/20,1)*0.5 + min(cats/5,1)*0.3 + (0.2 if has_adv else 0)
    exp_sc = min(yrs/8,1)*0.5 + min(bullets/12,1)*0.3 + (0.2 if yrs else 0)
    edu_sc = 0.85 if has_deg else 0.40
    pres   = 0.0
    if contact.get("email"):    pres += 0.20
    if contact.get("linkedin"): pres += 0.15
    if contact.get("github"):   pres += 0.10
    if has_sum:                 pres += 0.20
    pres += 0.35 if 300 <= wc <= 850 else 0.15 if wc > 150 else 0
    imp_sc = (0.5 if has_met else 0) + min(len(verbs)/8, 0.5)

    ov    = sk_sc*0.30 + exp_sc*0.25 + edu_sc*0.15 + min(pres,1)*0.15 + imp_sc*0.15
    pct   = round(ov * 100, 1)
    grade = "A+" if pct>=85 else "A" if pct>=75 else "B+" if pct>=65 else "B" if pct>=55 else "C" if pct>=45 else "D"
    return {
        "overall":pct, "grade":grade,
        "skills_pct":round(sk_sc*100), "experience_pct":round(exp_sc*100),
        "education_pct":round(edu_sc*100), "presentation_pct":round(min(pres,1)*100),
        "impact_pct":round(imp_sc*100),
        "word_count":wc, "bullet_count":bullets, "has_metrics":has_met,
        "has_summary":has_sum, "years_exp":yrs, "impact_verbs":verbs[:8], "total_skills":total_sk,
    }

def get_suggestions(contact, skills, scores, text):
    s = []
    if not contact.get("linkedin"):      s.append(("🔴 High","Contact","Add your LinkedIn profile URL"))
    if not contact.get("github") and skills.get("Programming Languages"):
                                          s.append(("🔴 High","Contact","Add GitHub profile to showcase your code"))
    if not scores["has_summary"]:        s.append(("🔴 High","Structure","Add a 3-4 sentence professional summary"))
    if not scores["has_metrics"]:        s.append(("🔴 High","Impact","Quantify achievements: 'Reduced latency by 40%', '$2M revenue'"))
    if scores["word_count"] < 250:       s.append(("🔴 High","Length","Resume too short — expand experience descriptions"))
    if scores["word_count"] > 950:       s.append(("🟡 Medium","Length","Resume may be too long — aim for 1-2 pages"))
    if not contact.get("name"):          s.append(("🔴 High","Contact","Ensure your full name appears at the top"))
    if not skills.get("ML / AI") and len(skills)>2:
                                          s.append(("🟡 Medium","Skills","Add ML/AI skills (TensorFlow, PyTorch, scikit-learn)"))
    if not skills.get("Cloud & DevOps"): s.append(("🟡 Medium","Skills","Add cloud/DevOps skills (AWS, Docker, Kubernetes)"))
    if not skills.get("Databases"):      s.append(("🟡 Medium","Skills","Add database skills (PostgreSQL, MongoDB, Redis)"))
    if scores["bullet_count"] < 5:       s.append(("🟡 Medium","Formatting","Use bullet points to describe achievements clearly"))
    s.append(("🟢 Low","Certifications","Add industry certs: AWS ML Specialty, GCP MLE, TF Developer"))
    return s

def match_job(resume_skills, jd_text):
    tl = jd_text.lower()
    jd_sk = set()
    for arr in SKILL_TAXONOMY.values():
        for s in arr:
            if re.search(r'\b' + _esc_re(s) + r'\b', tl, re.I):
                jd_sk.add(s)
    resume_flat = set(s for v in resume_skills.values() for s in v)
    matched = sorted(resume_flat & jd_sk)
    missing = sorted(jd_sk - resume_flat)[:10]
    pct = round(len(matched)/len(jd_sk)*100*0.7 + 30*0.3) if jd_sk else 50
    pct = min(pct, 99)
    rec = ("🎯 Excellent match! Apply with confidence." if pct>=80 else
           f"👍 Good match. Consider adding: {', '.join(missing[:2])}" if pct>=60 else
           f"⚠️ Moderate match. Key gaps: {', '.join(missing[:3])}" if pct>=40 else
           "❌ Low match. Focus on building the required skills first.")
    return {"pct":pct,"matched":matched,"missing":missing,"jd_count":len(jd_sk),"rec":rec}

# ── UI HELPERS ─────────────────────────────────────────────────────────────────
CHIP_COLORS = {
    "Programming Languages": ("#4af4e8","rgba(74,244,232,0.08)"),
    "ML / AI":               ("#f0476a","rgba(240,71,106,0.08)"),
    "Cloud & DevOps":        ("#7ef07c","rgba(126,240,124,0.08)"),
    "Data Engineering":      ("#a78bfa","rgba(167,139,250,0.08)"),
    "Databases":             ("#fb923c","rgba(251,146,60,0.08)"),
    "Visualization":         ("#f5c842","rgba(245,200,66,0.08)"),
    "Methods":               ("#94a3b8","rgba(148,163,184,0.08)"),
}

def chips(items, color, bg):
    sty = (f"display:inline-block;background:{bg};border:1px solid {color}33;"
           f"color:{color};border-radius:20px;padding:2px 10px;margin:2px;"
           f"font-size:11px;font-family:'JetBrains Mono',monospace")
    return "".join(f'<span style="{sty}">{s}</span>' for s in items)

def scol(v):
    return "#4af4e8" if v>=75 else "#7ef07c" if v>=55 else "#f5c842" if v>=40 else "#f0476a"

def gcol(g):
    return "#4af4e8" if g in("A+","A") else "#7ef07c" if g in("B+","B") else "#f5c842" if g=="C" else "#f0476a"

def bar(label, value):
    c = scol(value)
    return (f'<div style="margin:10px 0"><div style="display:flex;justify-content:space-between;'
            f'font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#6b7280;margin-bottom:5px">'
            f'<span>{label}</span><span style="color:{c};font-weight:600">{value}%</span></div>'
            f'<div style="height:5px;background:#1e2a3a;border-radius:3px">'
            f'<div style="height:100%;width:{value}%;background:{c};border-radius:3px"></div></div></div>')

def sug_row(priority, category, action):
    cc = {"🔴 High":"#f0476a","🟡 Medium":"#4af4e8","🟢 Low":"#374151"}.get(priority,"#374151")
    return (f'<div style="display:flex;align-items:flex-start;gap:10px;padding:9px 13px;'
            f'border-radius:8px;background:{cc}10;border-left:2px solid {cc}99;margin-bottom:5px">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;padding:2px 6px;'
            f'border-radius:3px;background:{cc}25;color:{cc};font-weight:700;letter-spacing:1px;'
            f'flex-shrink:0;margin-top:2px">{priority}</span>'
            f'<span style="font-size:12.5px;color:#94a3b8"><strong style="color:#e2e8f8">{category}:</strong> {action}</span></div>')

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="display:flex;align-items:center;gap:14px;padding:0 0 1.5rem 0;
            border-bottom:1px solid rgba(74,244,232,0.1);margin-bottom:1.5rem">
  <div style="width:36px;height:36px;border-radius:9px;background:linear-gradient(135deg,#4af4e8,#6366f1);
              display:flex;align-items:center;justify-content:center;font-size:18px">🤖</div>
  <div>
    <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:#e2e8f8;line-height:1">
      Resume<span style="color:#4af4e8">.</span>AI
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#4a5568;letter-spacing:2px">
      SMART RESUME ANALYZER v2.0
    </div>
  </div>
  <div style="margin-left:auto">
    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;padding:4px 12px;
                 border-radius:20px;border:1px solid rgba(126,240,124,0.3);
                 background:rgba(126,240,124,0.07);color:#7ef07c">● Live</span>
  </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:2px;color:#6b7280">UPLOAD RESUME FILE</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["pdf","docx","doc","txt","md"], help="PDF, DOCX, TXT — max 5 MB")
    st.markdown('<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:2px;color:#6b7280;margin-top:8px;display:block">OR PASTE RESUME TEXT</span>', unsafe_allow_html=True)
    resume_text = st.text_area("", height=260, key="rtxt",
        placeholder="John Doe\njohn@gmail.com | +91-9876543210\nlinkedin.com/in/johndoe | github.com/johndoe\n\nSummary\nSenior Data Scientist with 5+ years...\n\nExperience\nData Scientist | Google | 2020–Present\n• Built BERT model achieving 92% F1\n• Reduced latency by 60%\n\nSkills\nPython, TensorFlow, AWS, Docker, SQL")

with col2:
    st.markdown('<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:2px;color:#6b7280">JOB DESCRIPTION <span style="color:#374151">(OPTIONAL)</span></span>', unsafe_allow_html=True)
    jd_text = st.text_area("", height=360, key="jdtxt",
        placeholder="Paste job description here...\n\nWe are looking for a Senior Data Scientist:\n• Python, ML, deep learning required\n• NLP, BERT, transformers experience\n• AWS/GCP/Azure cloud experience\n• Docker, Kubernetes preferred\n• 3+ years production ML systems")

analyze_clicked = st.button("▶  Analyze Resume", use_container_width=True)

if analyze_clicked:
    text = ""
    if uploaded is not None:
        with st.spinner("📄 Parsing file..."):
            text = parse_document(uploaded.read(), uploaded.name)
        if len(text.strip()) < 30:
            st.error("⚠️ Could not extract text. If it's a PDF, ensure it's text-based (not scanned). Try pasting the text instead.")
            st.stop()
    elif resume_text.strip():
        text = resume_text.strip()
    else:
        st.error("❌ Please upload a file or paste resume text first.")
        st.stop()

    if len(text.strip()) < 30:
        st.error("❌ Resume text too short. Please provide more content.")
        st.stop()

    with st.spinner("🧠 Analyzing resume..."):
        contact = extract_contact(text)
        skills  = extract_skills(text)
        scores  = score_resume(text, contact, skills)
        sugs    = get_suggestions(contact, skills, scores, text)
        jm      = match_job(skills, jd_text) if jd_text.strip() else None

    st.session_state["result"] = {"contact":contact,"skills":skills,"scores":scores,"suggestions":sugs,"job_match":jm}
    st.success("✅ Analysis complete!")

if "result" in st.session_state:
    r = st.session_state["result"]
    s, c, sk, sugs, jm = r["scores"], r["contact"], r["skills"], r["suggestions"], r["job_match"]

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "🎯 Job Match", "🔍 Skills Breakdown", "🧠 Model Info"])

    with tab1:
        gc = gcol(s["grade"])
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(74,244,232,0.08),rgba(99,102,241,0.05));
                    border:1px solid rgba(74,244,232,0.18);border-radius:14px;padding:2rem;
                    text-align:center;margin-bottom:1.5rem">
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:3px;color:#4a5568;margin-bottom:10px">RESUME SCORE</div>
          <div style="font-family:'Syne',sans-serif;font-size:5rem;font-weight:800;color:{gc};line-height:1">{s['overall']}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;font-weight:700;color:{gc};margin-top:6px">{s['grade']}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#4a5568;margin-top:8px;letter-spacing:2px">OUT OF 100 POINTS</div>
        </div>""", unsafe_allow_html=True)

        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Skills",    s["total_skills"])
        m2.metric("Categories",len(sk))
        m3.metric("Yrs Exp",   s["years_exp"] or "—")
        m4.metric("Words",     s["word_count"])
        m5.metric("Bullets",   s["bullet_count"])
        m6.metric("Metrics",   "✓" if s["has_metrics"] else "✗")

        if c:
            chips_html = " ".join(
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                f'background:rgba(74,244,232,0.07);border:1px solid rgba(74,244,232,0.2);'
                f'color:#4af4e8;border-radius:6px;padding:3px 12px;margin:2px;display:inline-block">'
                f'<span style="color:#4a5568">{k}:</span> {v}</span>'
                for k,v in c.items() if v
            )
            st.markdown("**👤 Contact Detected**")
            st.markdown(chips_html, unsafe_allow_html=True)
            st.markdown("")

        lc, rc = st.columns(2)
        with lc:
            bars_html = "".join(bar(l,s[k]) for l,k in [("Skills","skills_pct"),("Experience","experience_pct"),("Education","education_pct"),("Presentation","presentation_pct"),("Impact","impact_pct")])
            st.markdown(f'<div style="background:#111827;border:1px solid rgba(74,244,232,0.1);border-radius:10px;padding:1.2rem"><div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:2px;color:#4a5568;text-transform:uppercase;margin-bottom:14px">Score Breakdown</div>{bars_html}</div>', unsafe_allow_html=True)
        with rc:
            sug_html = "".join(sug_row(*sg) for sg in sugs[:8])
            st.markdown(f'<div style="background:#111827;border:1px solid rgba(74,244,232,0.1);border-radius:10px;padding:1.2rem"><div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:2px;color:#4a5568;text-transform:uppercase;margin-bottom:14px">Improvements ({len(sugs)})</div>{sug_html}</div>', unsafe_allow_html=True)

        st.markdown("**🛠️ Skills Detected**")
        if sk:
            for cat, cat_skills in sk.items():
                color, bg = CHIP_COLORS.get(cat, ("#94a3b8","rgba(148,163,184,0.08)"))
                st.markdown(f'<div style="margin-bottom:12px"><div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:#4a5568;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">{cat}</div>{chips(cat_skills,color,bg)}</div>', unsafe_allow_html=True)
        else:
            st.info("No skills detected. Check resume formatting.")

    with tab2:
        if jm:
            mc = scol(jm["pct"])
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(126,240,124,0.08),rgba(74,244,232,0.05));
                        border:1px solid rgba(126,240,124,0.18);border-radius:14px;
                        padding:2rem;text-align:center;margin-bottom:1.5rem">
              <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:3px;color:#4a5568;margin-bottom:10px">JOB MATCH SCORE</div>
              <div style="font-family:'Syne',sans-serif;font-size:4.5rem;font-weight:800;color:{mc};line-height:1">{jm['pct']}%</div>
              <div style="font-size:13px;color:#6b7280;margin-top:12px;max-width:460px;margin-left:auto;margin-right:auto;line-height:1.7">{jm['rec']}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#374151;margin-top:10px">{jm['jd_count']} required skills found in JD</div>
            </div>""", unsafe_allow_html=True)
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown(f"**✅ Matched ({len(jm['matched'])})**")
                if jm["matched"]: st.markdown(chips(jm["matched"],"#7ef07c","rgba(126,240,124,0.08)"), unsafe_allow_html=True)
                else: st.caption("No direct matches found.")
            with mc2:
                st.markdown(f"**❌ Missing — Learn These ({len(jm['missing'])})**")
                if jm["missing"]: st.markdown(chips(jm["missing"],"#f0476a","rgba(240,71,106,0.08)"), unsafe_allow_html=True)
                else: st.success("No major skill gaps!")
        else:
            st.info("📋 Paste a job description above and re-analyze to see your match score.")

    with tab3:
        st.markdown(f"**{s['total_skills']} unique skills across {len(sk)} categories**")
        for cat, cat_skills in sk.items():
            color, bg = CHIP_COLORS.get(cat, ("#94a3b8","rgba(148,163,184,0.08)"))
            with st.expander(f"{cat}  ·  {len(cat_skills)} skills", expanded=True):
                st.markdown(chips(cat_skills, color, bg), unsafe_allow_html=True)
        st.divider()
        st.markdown("**Missing from your resume:**")
        present = set(s for v in sk.values() for s in v)
        for cat, all_sk in SKILL_TAXONOMY.items():
            miss = [s for s in all_sk if s not in present]
            if miss:
                st.caption(cat)
                miss_chips = "".join(f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;background:rgba(255,255,255,0.02);border:1px solid #1e2a3a;color:#374151;border-radius:20px;padding:2px 9px;margin:2px;text-decoration:line-through;display:inline-block">{s}</span>' for s in miss)
                st.markdown(miss_chips, unsafe_allow_html=True)

    with tab4:
        st.markdown("""
        <div style="background:#111827;border:1px solid rgba(74,244,232,0.1);border-radius:10px;padding:1.3rem;margin-bottom:1rem">
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:2px;color:#4a5568;text-transform:uppercase;margin-bottom:14px">Analysis Engine</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;line-height:2.2;color:#6b7280">
            <div><span style="color:#4af4e8">Type      </span> Rule-based NER + Taxonomy skill extraction</div>
            <div><span style="color:#4af4e8">Skills    </span> 7 categories, 150+ skills in taxonomy</div>
            <div><span style="color:#4af4e8">Contact   </span> Regex: email, phone, LinkedIn, GitHub, name</div>
            <div><span style="color:#4af4e8">Scoring   </span> skills 30% · exp 25% · edu 15% · presentation 15% · impact 15%</div>
            <div><span style="color:#4af4e8">Runtime   </span> Zero ML training — instant startup on Streamlit Cloud</div>
          </div>
        </div>""", unsafe_allow_html=True)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Skills in DB","150+"); m2.metric("Categories","7"); m3.metric("Scoring Factors","5"); m4.metric("Startup","< 2s")
        st.markdown("**Upgrade Path**")
        for title, desc in [("bert-base-uncased NER","Token-classification, F1 ~95%"),("Kaggle Resume Dataset","2,400 real labeled resumes"),("sentence-transformers","Semantic job matching"),("Tesseract OCR","Scanned PDF support"),("spaCy pipeline","Production-grade NER")]:
            st.markdown(f'<div style="display:flex;gap:12px;padding:9px 0;border-bottom:1px solid #1e2a3a;font-size:12.5px;color:#6b7280"><span style="color:#4af4e8;font-family:\'JetBrains Mono\',monospace;flex-shrink:0">→</span><span><strong style="color:#e2e8f8">{title}</strong> — {desc}</span></div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#374151;font-family:'JetBrains Mono',monospace">
      <div style="font-size:3.5rem;margin-bottom:16px;opacity:0.15">📄</div>
      <div style="font-size:13px;color:#4a5568">Upload a resume or paste text above, then click <strong style="color:#4af4e8">Analyze Resume</strong></div>
      <div style="font-size:11px;color:#374151;margin-top:8px">Supports PDF · DOCX · TXT · Markdown</div>
    </div>""", unsafe_allow_html=True)
