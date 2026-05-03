"""
Full Resume Analyzer Pipeline v2
Integrates: Trained NER Model + Rule-based scoring + Job Matching + PDF/DOCX parsing

FIXES vs original:
1. sys.path patched so document_parser and ner_model_trainer are always importable.
2. NERExtractor._load() now validates the pickle has all required keys before using them.
3. NERExtractor.extract() handles the edge case where feature_names is None (untrained model).
4. Orphan I- tags (no preceding B-) are treated as B- instead of being silently dropped.
5. ResumeScorer.score() — 'education' score key was 'education' in weights but code
   used 'education_pct' only in the return dict; both now consistent.
6. analyze_file() path injection uses insert(0,...) so project root always wins.
7. All defaultdict usages now import cleanly at top level.
"""

import re
import json
import os
import sys
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ── Path setup: ensure both the project root and models/ dir are on sys.path ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in [_HERE, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


SKILL_TAXONOMY = {
    "programming_languages": [
        "python","java","javascript","typescript","c++","c#","go","rust",
        "kotlin","swift","r","scala","ruby","php","sql","bash","html","css","matlab"
    ],
    "ml_ai": [
        "machine learning","deep learning","nlp","natural language processing",
        "computer vision","bert","gpt","transformers","huggingface","tensorflow",
        "pytorch","keras","scikit-learn","xgboost","lightgbm","catboost",
        "reinforcement learning","llm","openai","langchain","yolo","resnet","lstm","rnn","cnn"
    ],
    "data_engineering": [
        "apache spark","kafka","airflow","hadoop","dbt","etl","data pipeline",
        "snowflake","databricks","pyspark","pandas","numpy","dask","hive","flink"
    ],
    "cloud_devops": [
        "aws","azure","gcp","google cloud","docker","kubernetes","terraform",
        "ansible","ci/cd","jenkins","github actions","linux","microservices",
        "fastapi","flask","django","spring boot","grpc","prometheus","grafana","mlflow","mlops"
    ],
    "databases": [
        "postgresql","mysql","mongodb","redis","cassandra","elasticsearch",
        "bigquery","redshift","dynamodb","firebase","neo4j","sqlite","oracle"
    ],
    "data_visualization": [
        "tableau","power bi","matplotlib","seaborn","plotly","d3.js",
        "looker","grafana","bokeh","dash","looker studio"
    ],
    "soft_skills": [
        "agile","scrum","team leadership","project management","communication",
        "problem solving","mentoring","a/b testing","statistics","hypothesis testing"
    ],
}

IMPACT_VERBS = [
    "built","developed","designed","created","architected","engineered","led","managed",
    "improved","optimized","reduced","increased","deployed","launched","delivered",
    "implemented","automated","scaled","shipped","accelerated","collaborated",
    "analyzed","researched","established"
]

_REQUIRED_PICKLE_KEYS = {'clf', 'label_encoder', 'feature_names', 'trained'}


# ── NER Extractor ─────────────────────────────────────────────────────────────

class NERExtractor:
    """Wrapper around the trained NER model pickle."""

    def __init__(self, model_path: str):
        self.loaded = False
        self.clf = None
        self.le = None
        self.feature_names = None
        if model_path and os.path.exists(model_path):
            self._load(model_path)
        else:
            print(f"[NERExtractor] Model not found at '{model_path}'. Using rule-based fallback only.")

    def _load(self, path: str):
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            # FIX: validate all required keys are present before unpacking
            missing = _REQUIRED_PICKLE_KEYS - set(state.keys())
            if missing:
                raise KeyError(f"Pickle is missing keys: {missing}")
            self.clf = state['clf']
            self.le = state['label_encoder']
            self.feature_names = state['feature_names']
            self.loaded = bool(state['trained']) and self.feature_names is not None
            print(f"[NERExtractor] Loaded model from {path} ({len(self.feature_names)} features)")
        except Exception as e:
            print(f"[NERExtractor] Load failed: {e}. Using rule-based fallback.")
            self.loaded = False

    def extract(self, text: str) -> Dict[str, List[str]]:
        if not self.loaded or self.feature_names is None:
            return {}
        try:
            try:
                from ner_model_trainer import FeatureExtractor, tokenize
            except ImportError:
                from models.ner_model_trainer import FeatureExtractor, tokenize

            fe = FeatureExtractor()
            tokens = tokenize(text)
            if not tokens:
                return {}
            token_strs = [t[0] for t in tokens]
            feats = fe.extract_sequence(token_strs)
            X = np.array(
                [[f.get(fn, 0.0) for fn in self.feature_names] for f in feats],
                dtype=np.float32
            )
            y_enc = self.clf.predict(X)
            y_labels = self.le.inverse_transform(y_enc)
        except Exception as e:
            print(f"[NERExtractor] Inference error: {e}")
            return {}

        entities: Dict[str, list] = defaultdict(list)
        current_toks = []
        current_label = None

        for (tok, _, _), label in zip(tokens, y_labels):
            if label.startswith("B-"):
                if current_toks and current_label:
                    entities[current_label].append(" ".join(current_toks))
                current_toks = [tok]
                current_label = label[2:]
            elif label.startswith("I-"):
                label_type = label[2:]
                if current_label == label_type and current_toks:
                    # Continuation
                    current_toks.append(tok)
                else:
                    # FIX: orphan I- tag — treat as B-
                    if current_toks and current_label:
                        entities[current_label].append(" ".join(current_toks))
                    current_toks = [tok]
                    current_label = label_type
            else:
                if current_toks and current_label:
                    entities[current_label].append(" ".join(current_toks))
                current_toks = []
                current_label = None

        if current_toks and current_label:
            entities[current_label].append(" ".join(current_toks))

        return {k: list(dict.fromkeys(v)) for k, v in entities.items()}


# ── Rule-based Extractor ──────────────────────────────────────────────────────

class RuleBasedExtractor:
    """Fast regex + taxonomy-based extractor. Always runs; supplements NER."""

    def extract_contact(self, text: str) -> Dict:
        c = {}
        em = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        if em:
            c['email'] = em.group()
        ph = re.search(r'(\+?[\d][\d\s\-\(\)\.]{8,14}\d)', text)
        if ph:
            c['phone'] = ph.group().strip()
        li = re.search(r'linkedin\.com/in/[^\s\n,)]+', text, re.I)
        if li:
            c['linkedin'] = li.group()
        gh = re.search(r'github\.com/[^\s\n,)]+', text, re.I)
        if gh:
            c['github'] = gh.group()
        # Name: first non-empty line of 2–5 words, all alpha/space/dash
        for line in text.split('\n')[:6]:
            line = line.strip()
            if 3 < len(line) < 55 and re.match(r'^[A-Za-z][\w\s\-\.]+$', line):
                words = line.split()
                if 2 <= len(words) <= 5 and all(w[0].isupper() for w in words if w.isalpha()):
                    c['name'] = line
                    break
        return c

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        text_lower = text.lower()
        found: Dict[str, list] = defaultdict(list)
        seen: set = set()
        for cat, skills in SKILL_TAXONOMY.items():
            for skill in sorted(skills, key=len, reverse=True):
                pat = r'\b' + re.escape(skill) + r'\b'
                if re.search(pat, text_lower) and skill not in seen:
                    found[cat].append(skill)
                    seen.add(skill)
        return dict(found)

    def extract_years_exp(self, text: str) -> Optional[int]:
        m = re.search(r'(\d+)\+?\s*(?:years?|yrs?)[\s\w]{0,10}(?:experience|exp)', text, re.I)
        return int(m.group(1)) if m else None

    def extract_education(self, text: str) -> List[Dict]:
        edu = []
        for m in re.finditer(
            r'(b\.?tech|m\.?tech|b\.?s\.?|m\.?s\.?|ph\.?d|mba|b\.?e\.?|m\.?e\.|bachelor|master|doctorate)',
            text, re.I
        ):
            edu.append({'degree': m.group().upper().replace('.', '').replace(' ', '')})
        return edu[:3]

    def extract_impact(self, text: str) -> Dict:
        text_lower = text.lower()
        has_metrics = bool(re.search(r'\d+%|\$\d+|\d+x|\d+\s*(million|thousand|\bk\b)', text, re.I))
        verbs_found = [v for v in IMPACT_VERBS if re.search(r'\b' + v + r'\b', text_lower)]
        bullets = re.findall(r'[•\-\*]\s*(.+)', text)
        return {
            'has_quantified_impact': has_metrics,
            'impact_verbs': verbs_found[:10],
            'bullet_count': len(bullets),
            'word_count': len(text.split()),
            'has_summary': bool(re.search(r'\b(summary|profile|objective|about)\b', text, re.I)),
        }


# ── Scorer ────────────────────────────────────────────────────────────────────

class ResumeScorer:
    WEIGHTS = {'skills': 0.30, 'experience': 0.25, 'education': 0.15, 'presentation': 0.15, 'impact': 0.15}

    def score(self, parsed: Dict) -> Dict:
        s = {}

        # Skills
        total = sum(len(v) for v in parsed['skills'].values())
        cats = len(parsed['skills'])
        has_adv = bool(parsed['skills'].get('ml_ai') or parsed['skills'].get('cloud_devops'))
        s['skills'] = min(total / 20, 1) * 0.5 + min(cats / 5, 1) * 0.3 + (0.2 if has_adv else 0)

        # Experience
        yrs = parsed.get('years_experience') or 0
        bullets = parsed.get('impact', {}).get('bullet_count', 0)
        s['experience'] = min(yrs / 8, 1) * 0.5 + min(bullets / 12, 1) * 0.3 + (0.2 if yrs > 0 else 0)

        # Education
        edu = parsed.get('education', [])
        deg_map = {
            'PHD': 1.0, 'MS': 0.85, 'ME': 0.85, 'MTECH': 0.85, 'MBA': 0.85,
            'BS': 0.7, 'BE': 0.7, 'BTECH': 0.7, 'BSC': 0.65, 'MSC': 0.85,
        }
        best = 0.4
        for e in edu:
            deg_clean = e.get('degree', '').upper().replace('.', '').replace(' ', '')
            for k, v in deg_map.items():
                if k in deg_clean:
                    best = max(best, v)
        s['education'] = best

        # Presentation
        c = parsed.get('contact', {})
        p = 0.0
        p += 0.2 if c.get('email') else 0
        p += 0.15 if c.get('linkedin') else 0
        p += 0.1 if c.get('github') else 0
        p += 0.2 if parsed.get('impact', {}).get('has_summary') else 0
        wc = parsed.get('impact', {}).get('word_count', 0)
        p += 0.35 if 300 <= wc <= 800 else (0.15 if wc > 150 else 0)
        s['presentation'] = min(p, 1.0)

        # Impact
        imp = parsed.get('impact', {})
        iv = min(len(imp.get('impact_verbs', [])) / 8, 0.5)
        s['impact'] = (0.5 if imp.get('has_quantified_impact') else 0) + iv

        overall = sum(s[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        pct = round(overall * 100, 1)
        grade = ('A+' if pct >= 85 else 'A' if pct >= 75 else 'B+' if pct >= 65
                 else 'B' if pct >= 55 else 'C' if pct >= 45 else 'D')

        return {
            'overall': pct,
            'grade': grade,
            'skills_pct': round(s['skills'] * 100),
            'experience_pct': round(s['experience'] * 100),
            'education_pct': round(s['education'] * 100),
            'presentation_pct': round(s['presentation'] * 100),
            'impact_pct': round(s['impact'] * 100),
        }

    def match_job(self, parsed: Dict, jd: str) -> Dict:
        jd_lower = jd.lower()
        resume_skills = set(s for skills in parsed['skills'].values() for s in skills)
        jd_skills = set()
        for skills in SKILL_TAXONOMY.values():
            for s in skills:
                if re.search(r'\b' + re.escape(s) + r'\b', jd_lower):
                    jd_skills.add(s)
        matched = resume_skills & jd_skills
        missing = sorted(jd_skills - resume_skills)[:10]
        skill_pct = round(len(matched) / len(jd_skills) * 100) if jd_skills else 0
        final = min(round(skill_pct * 0.7 + 30 * 0.3), 99)
        if not jd_skills:
            final = 50
        rec = (
            "Excellent match! Apply with confidence." if final >= 80 else
            f"Good match. Consider adding: {', '.join(missing[:2])}" if final >= 60 else
            f"Moderate match. Key gaps: {', '.join(missing[:3])}." if final >= 40 else
            "Low match. Focus on required skills first."
        )
        return {
            'match_percentage': final,
            'matched_skills': sorted(matched),
            'missing_skills': missing,
            'recommendation': rec,
            'jd_skills_found': len(jd_skills),
        }


# ── Suggestion Engine ─────────────────────────────────────────────────────────

class SuggestionEngine:
    def generate(self, parsed: Dict, scores: Dict) -> List[Dict]:
        s = []
        c = parsed.get('contact', {})
        if not c.get('linkedin'):
            s.append({'priority': 'high', 'category': 'Contact', 'action': 'Add your LinkedIn profile URL'})
        if not c.get('github') and parsed['skills'].get('programming_languages'):
            s.append({'priority': 'high', 'category': 'Contact', 'action': 'Add GitHub profile to showcase code'})
        if not parsed.get('impact', {}).get('has_summary'):
            s.append({'priority': 'high', 'category': 'Structure', 'action': 'Add a 3-4 sentence professional summary'})
        if not parsed.get('impact', {}).get('has_quantified_impact'):
            s.append({'priority': 'high', 'category': 'Impact',
                      'action': "Add metrics: 'Reduced latency by 40%', '$2M revenue impact'"})
        if not parsed['skills'].get('ml_ai') and len(parsed['skills']) > 2:
            s.append({'priority': 'medium', 'category': 'Skills',
                      'action': 'Add ML/AI skills (TensorFlow, PyTorch, scikit-learn)'})
        if not parsed['skills'].get('cloud_devops'):
            s.append({'priority': 'medium', 'category': 'Skills',
                      'action': 'Add cloud experience (AWS/GCP/Azure/Docker)'})
        wc = parsed.get('impact', {}).get('word_count', 0)
        if wc < 250:
            s.append({'priority': 'high', 'category': 'Length',
                      'action': 'Resume too short — expand experience descriptions'})
        elif wc > 900:
            s.append({'priority': 'medium', 'category': 'Length',
                      'action': 'Resume may be too long — aim for 1-2 pages'})
        if not c.get('name'):
            s.append({'priority': 'high', 'category': 'Contact',
                      'action': 'Ensure your name appears clearly at the top'})
        s.append({'priority': 'low', 'category': 'Certifications',
                  'action': 'Add industry certs: AWS, GCP, TF Developer, etc.'})
        return sorted(s, key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['priority']])


# ── Main Pipeline ─────────────────────────────────────────────────────────────

class ResumeAnalyzerV2:
    """
    Full production pipeline:
    Trained NER + Rule-based extraction + Scoring + Job Matching + Suggestions
    """

    def __init__(self, ner_model_path: str = None):
        self.ner = NERExtractor(ner_model_path) if ner_model_path else None
        self.rules = RuleBasedExtractor()
        self.scorer = ResumeScorer()
        self.suggester = SuggestionEngine()

    def analyze(self, text: str, job_description: str = "") -> Dict:
        # 1. Rule-based contact (most reliable for structured fields)
        contact = self.rules.extract_contact(text)

        # 2. Taxonomy-based skill extraction (high precision)
        skills = self.rules.extract_skills(text)

        # 3. Impact metadata
        impact = self.rules.extract_impact(text)
        years_exp = self.rules.extract_years_exp(text)
        education = self.rules.extract_education(text)

        # 4. Augment with NER model predictions
        ner_entities: Dict = {}
        if self.ner and self.ner.loaded:
            ner_entities = self.ner.extract(text)
            if not contact.get('name') and ner_entities.get('PERSON'):
                contact['name'] = ner_entities['PERSON'][0]
            if ner_entities.get('COMPANY'):
                ner_entities['companies'] = ner_entities.pop('COMPANY')
            if ner_entities.get('JOB_TITLE'):
                ner_entities['job_titles'] = ner_entities.pop('JOB_TITLE')

        parsed = {
            'contact': contact,
            'skills': skills,
            'education': education,
            'impact': impact,
            'years_experience': years_exp,
            'ner_entities': ner_entities,
            'total_skills': sum(len(v) for v in skills.values()),
            'skill_categories': list(skills.keys()),
        }

        scores = self.scorer.score(parsed)
        suggestions = self.suggester.generate(parsed, scores)

        result = {'parsed': parsed, 'scores': scores, 'suggestions': suggestions}
        if job_description.strip():
            result['job_match'] = self.scorer.match_job(parsed, job_description)

        return result

    def analyze_file(self, file_bytes: bytes, filename: str, job_description: str = "") -> Dict:
        # FIX: Ensure project root is FIRST in sys.path for document_parser import
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)

        try:
            from document_parser import DocumentParser
        except ImportError:
            raise ImportError(
                "document_parser.py not found. Ensure it is in the same directory as pipeline_v2.py."
            )

        text = DocumentParser.parse(file_bytes, filename)
        if len(text.strip()) < 30:
            return {'error': 'Could not extract text. Ensure the file is a text-based PDF or DOCX (not scanned).'}
        result = self.analyze(text, job_description)
        result['source_file'] = filename
        result['extracted_chars'] = len(text)
        return result

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ResumeAnalyzerV2":
        with open(path, 'rb') as f:
            return pickle.load(f)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ner_path = os.path.join(_HERE, "ner_model.pkl")
    if not os.path.exists(ner_path):
        ner_path = os.path.join(_ROOT, "models", "ner_model.pkl")

    pipeline = ResumeAnalyzerV2(ner_model_path=ner_path)

    sample = """
Priya Sharma
priya.sharma99@gmail.com | +91-9876543210 | linkedin.com/in/priya-sharma | github.com/priyasharma

Summary
Senior Data Scientist with 5+ years of experience in Python, TensorFlow, NLP, and MLOps.
Proven track record of deploying ML models that increased revenue by $2M.

Experience
Senior Data Scientist | Google | Jan 2020 – Present
• Built BERT-based resume classification model achieving 92% F1 score
• Reduced model inference latency by 60% through ONNX optimization
• Led team of 4 to deploy recommendation engine serving 10M+ users

Data Scientist | Flipkart | Mar 2018 – Dec 2019
• Developed customer churn model achieving 89% AUC using XGBoost
• Built Kafka + Spark pipeline processing 2TB+ daily data

Education
M.Tech in Computer Science | IIT Bombay | 2018
GPA: 9.1/10

Skills
Python, TensorFlow, PyTorch, scikit-learn, BERT, NLP, SQL, PostgreSQL, AWS, Docker, Kubernetes, Kafka, Apache Spark, MLflow, FastAPI

Certifications
AWS Certified Machine Learning Specialty
Google Professional Data Engineer
"""

    jd = """
We need a Senior Data Scientist with: Python, deep learning, NLP, transformers, PyTorch, AWS,
Docker, Kubernetes, SQL, MLOps, 4+ years experience, production ML systems.
"""

    print("Running full analysis...")
    result = pipeline.analyze(sample, jd)
    print(f"\nOverall Score: {result['scores']['overall']}/100  Grade: {result['scores']['grade']}")
    print(f"Skills Found: {result['parsed']['total_skills']} across {len(result['parsed']['skill_categories'])} categories")
    print(f"Contact: {result['parsed']['contact']}")
    if 'job_match' in result:
        jm = result['job_match']
        print(f"Job Match: {jm['match_percentage']}%  —  {jm['recommendation']}")
    print(f"\nTop 3 Suggestions:")
    for sug in result['suggestions'][:3]:
        print(f"  [{sug['priority'].upper()}] {sug['action']}")
