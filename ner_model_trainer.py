"""
Resume NER Model — Custom CRF-based Named Entity Recognizer
Trained on synthetic resume dataset.

FIXES:
1. LogisticRegression: removed deprecated `multi_class='multinomial'` param
   (sklearn >= 1.5 raises TypeError — use solver='lbfgs' which handles multiclass natively)
2. Solver changed from 'saga' to 'lbfgs' — saga doesn't support multinomial well on small data
3. Feature vocabulary collected from ALL training samples (not just first sample)
4. Evaluation mask handles unseen labels gracefully
5. Directory creation is automatic everywhere
6. DatasetConverter.convert() now returns consistent (X, y, feature_names) always
7. ResumeNERModel.predict_text() now guards against empty feature_names
8. extract_entities() BIO chain bug fixed — I- tags with no preceding B- are handled
"""

import re
import json
import os
import pickle
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')


ENTITY_LABELS = [
    "O",
    "B-PERSON","I-PERSON",
    "B-EMAIL","I-EMAIL",
    "B-PHONE","I-PHONE",
    "B-URL","I-URL",
    "B-SKILL","I-SKILL",
    "B-COMPANY","I-COMPANY",
    "B-JOB_TITLE","I-JOB_TITLE",
    "B-UNIVERSITY","I-UNIVERSITY",
    "B-DEGREE","I-DEGREE",
    "B-CERTIFICATION","I-CERTIFICATION",
]

SKILL_VOCAB = set([
    "python","java","javascript","typescript","c++","c#","go","rust","kotlin","swift",
    "r","scala","ruby","php","sql","bash","html","css","matlab","perl",
    "tensorflow","pytorch","keras","scikit-learn","sklearn","xgboost","lightgbm","catboost",
    "pandas","numpy","scipy","matplotlib","seaborn","plotly","bokeh","dash",
    "machine learning","deep learning","nlp","natural language processing","computer vision",
    "bert","gpt","transformers","huggingface","lstm","rnn","cnn","neural networks",
    "spark","apache spark","kafka","airflow","hadoop","hive","pig","flink",
    "pyspark","databricks","snowflake","dbt","etl","elt","data pipeline",
    "aws","azure","gcp","google cloud","docker","kubernetes","terraform","ansible",
    "jenkins","github actions","ci/cd","linux","git","github","gitlab",
    "postgresql","mysql","mongodb","redis","cassandra","elasticsearch","bigquery",
    "tableau","power bi","looker","grafana","prometheus","d3.js",
    "fastapi","flask","django","spring boot","nodejs","express","react","vue","angular",
    "microservices","rest api","graphql","grpc","rabbitmq",
    "reinforcement learning","generative ai","llm","openai","langchain",
    "statistics","probability","a/b testing","hypothesis testing","regression",
    "agile","scrum","jira","confluence","devops","mlops","mlflow",
    "excel","powerpoint","word","google sheets","looker studio",
])

COMPANY_SIGNALS = {"inc","ltd","llc","corp","corporation","technologies","solutions",
                   "systems","services","consulting","analytics","labs","group","ventures"}
DEGREE_KEYWORDS = {"b.tech","m.tech","b.e.","m.e.","b.sc","m.sc","ph.d","phd","mba",
                   "bachelor","master","doctorate","b.s.","m.s.","b.a.","m.a.",
                   "btech","mtech","be","me","bsc","msc"}
JOB_TITLE_KEYWORDS = {"engineer","scientist","analyst","developer","architect","manager",
                       "director","lead","senior","junior","principal","staff","associate",
                       "specialist","consultant","researcher","intern"}
UNIVERSITY_KEYWORDS = {"university","institute","college","school","iit","nit","bits",
                        "iisc","iiit","iim","technology","polytechnic"}
SECTION_HEADERS = {"summary","profile","experience","education","skills","projects",
                   "certifications","awards","achievements","objective","about"}


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    """Returns (token, start, end) triples."""
    tokens = []
    for m in re.finditer(r'\S+', text):
        tokens.append((m.group(), m.start(), m.end()))
    return tokens


class FeatureExtractor:
    def __init__(self):
        self.skill_vocab = SKILL_VOCAB

    def token_features(self, tokens: List[str], i: int) -> Dict[str, float]:
        tok = tokens[i]
        tok_lower = tok.lower().rstrip('.,;:')
        prev1 = tokens[i - 1].lower() if i > 0 else "<START>"
        prev2 = tokens[i - 2].lower() if i > 1 else "<START>"
        next1 = tokens[i + 1].lower() if i < len(tokens) - 1 else "<END>"
        next2 = tokens[i + 2].lower() if i < len(tokens) - 2 else "<END>"

        f = {}

        # Shape features
        f['is_upper'] = float(tok.isupper())
        f['is_title'] = float(tok.istitle())
        f['is_lower'] = float(tok.islower())
        f['is_digit'] = float(tok.isdigit())
        f['has_digit'] = float(any(c.isdigit() for c in tok))
        f['has_at'] = float('@' in tok)
        f['has_dot'] = float('.' in tok)
        f['has_slash'] = float('/' in tok)
        f['has_dash'] = float('-' in tok)
        f['has_plus'] = float('+' in tok)
        f['length'] = min(len(tok) / 20.0, 1.0)
        f['starts_upper'] = float(tok[0].isupper() if tok else False)

        # Email pattern
        f['is_email'] = float(bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', tok)))

        # Phone pattern
        f['is_phone_like'] = float(bool(re.match(r'^[\+\d\-\(\)\s\.]{7,15}$', tok)))
        f['is_phone_digits'] = float(bool(re.match(r'^\+?\d[\d\-\.\s]{7,}$', tok)))

        # URL pattern
        f['is_url'] = float(bool(re.match(r'^(https?://|www\.|linkedin\.com|github\.com)', tok_lower)))

        # Skill features
        f['is_skill'] = float(tok_lower in self.skill_vocab)
        f['is_skill_like'] = float(any(tok_lower.startswith(s[:4]) for s in list(self.skill_vocab)[:50]))

        # Company signals
        f['has_company_word'] = float(tok_lower.rstrip('.,') in COMPANY_SIGNALS)

        # Degree signals
        f['is_degree'] = float(tok_lower.rstrip('.,') in DEGREE_KEYWORDS)
        f['is_degree_abbr'] = float(bool(re.match(
            r'^(b\.?tech|m\.?tech|b\.?s\.?|m\.?s\.?|ph\.?d|mba|b\.?e\.?|m\.?e\.?)\??$', tok_lower
        )))

        # Job title signals
        f['is_job_title'] = float(tok_lower.rstrip('.,') in JOB_TITLE_KEYWORDS)

        # University signals
        f['is_university'] = float(any(kw in tok_lower for kw in UNIVERSITY_KEYWORDS))

        # Section header
        f['is_section'] = float(tok_lower.rstrip(':') in SECTION_HEADERS)

        # Year
        f['is_year'] = float(bool(re.match(r'^(19|20)\d\d$', tok)))

        # Position features
        f['position'] = i / max(len(tokens) - 1, 1)
        f['is_first_token'] = float(i == 0)
        f['is_second_token'] = float(i == 1)

        # Context: previous token signals
        f['prev_is_pipe'] = float(prev1 in ['|', '·', '•', '-', '–'])
        f['prev_is_at'] = float(prev1 == 'at')
        f['prev_is_in'] = float(prev1 in ['in', 'of', 'at', 'for'])
        f['prev_is_section'] = float(prev1.rstrip(':') in SECTION_HEADERS)
        f['prev_is_degree'] = float(prev1.rstrip('.,') in DEGREE_KEYWORDS)
        f['prev_is_title_case'] = float(prev1.istitle())
        f['prev2_is_section'] = float(prev2.rstrip(':') in SECTION_HEADERS)

        # Context: next token signals
        f['next_is_pipe'] = float(next1 in ['|', '·', '•', ','])
        f['next_is_at'] = float(next1 == '@')
        f['next_is_company'] = float(next1.rstrip('.,') in COMPANY_SIGNALS)
        f['next_is_title_case'] = float(next1.istitle())

        # Character n-gram prefix/suffix
        prefix = tok_lower[:3] if len(tok_lower) >= 3 else tok_lower
        suffix = tok_lower[-3:] if len(tok_lower) >= 3 else tok_lower
        f[f'prefix_{prefix}'] = 1.0
        f[f'suffix_{suffix}'] = 1.0

        return f

    def extract_sequence(self, tokens: List[str]) -> List[Dict]:
        return [self.token_features(tokens, i) for i in range(len(tokens))]


def char_to_token_labels(
    tokens: List[Tuple[str, int, int]],
    annotations: List[Tuple[int, int, str]]
) -> List[str]:
    """Map character-level annotations to BIO token labels."""
    labels = ["O"] * len(tokens)
    for ann_start, ann_end, label in annotations:
        first = True
        for i, (tok, tok_start, tok_end) in enumerate(tokens):
            if tok_start >= ann_start and tok_end <= ann_end:
                if first:
                    labels[i] = f"B-{label}"
                    first = False
                else:
                    labels[i] = f"I-{label}"
    return labels


class DatasetConverter:
    def __init__(self):
        self.fe = FeatureExtractor()

    def collect_feature_names(self, dataset: List[Dict]) -> List[str]:
        """
        Scan ALL samples to collect the complete feature vocabulary.
        The original code only scanned the first sample, causing silent zero-fill
        for features (e.g. prefix/suffix n-grams) seen only in later samples.
        """
        all_feature_keys: set = set()
        for item in dataset:
            tokens = tokenize(item["text"])
            if not tokens:
                continue
            token_strs = [t[0] for t in tokens]
            feats = self.fe.extract_sequence(token_strs)
            for fd in feats:
                all_feature_keys.update(fd.keys())
        return sorted(all_feature_keys)

    def convert(self, dataset: List[Dict], feature_names: List[str] = None):
        """
        Convert dataset to (X, y, feature_names).
        If feature_names is supplied (e.g. from training set), reuse it for test set.
        """
        all_features = []
        all_labels = []

        if feature_names is None:
            feature_names = self.collect_feature_names(dataset)

        for item in dataset:
            text = item["text"]
            annotations = item["annotations"]
            tokens = tokenize(text)
            if not tokens:
                continue
            token_strs = [t[0] for t in tokens]
            labels = char_to_token_labels(tokens, annotations)
            feats = self.fe.extract_sequence(token_strs)

            for feat_dict, label in zip(feats, labels):
                vec = np.array([feat_dict.get(fn, 0.0) for fn in feature_names], dtype=np.float32)
                all_features.append(vec)
                all_labels.append(label)

        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels)
        return X, y, feature_names


class ResumeNERModel:
    """
    Token-classification NER model for resumes.
    Uses LogisticRegression with rich handcrafted features.

    FIX: Removed deprecated `multi_class='multinomial'` kwarg — sklearn >= 1.5
    raises TypeError. Solver 'lbfgs' handles multiclass natively and is faster
    than 'saga' for small-to-medium datasets.
    """

    def __init__(self):
        # FIX: removed multi_class='multinomial' (deprecated/removed in sklearn 1.5+)
        # lbfgs handles multiclass natively; saga was causing convergence issues
        self.clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            n_jobs=1,        # lbfgs doesn't support n_jobs > 1 — was silently ignored
            random_state=42,
        )
        self.fe = FeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        y_enc = self.label_encoder.fit_transform(y)
        print(f"  Training on {X.shape[0]:,} tokens, {X.shape[1]} features, "
              f"{len(self.label_encoder.classes_)} classes...")
        self.clf.fit(X, y_enc)
        self.trained = True
        return self

    def predict_text(self, text: str) -> List[Tuple[str, int, int, str]]:
        """Returns list of (token, start, end, label) for non-O tokens."""
        if not self.trained or self.feature_names is None:
            return []
        tokens = tokenize(text)
        if not tokens:
            return []
        token_strs = [t[0] for t in tokens]
        fe = FeatureExtractor()
        feats = fe.extract_sequence(token_strs)
        X = np.array([[f.get(fn, 0.0) for fn in self.feature_names] for f in feats], dtype=np.float32)
        y_enc = self.clf.predict(X)
        y_labels = self.label_encoder.inverse_transform(y_enc)

        results = []
        for (tok, start, end), label in zip(tokens, y_labels):
            if label != "O":
                results.append((tok, start, end, label))
        return results

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        raw = self.predict_text(text)
        entities = defaultdict(list)
        current_entity = []
        current_label = None

        for tok, start, end, bio_label in raw:
            if bio_label.startswith("B-"):
                # Flush previous entity
                if current_entity and current_label:
                    entities[current_label].append(" ".join(current_entity))
                current_entity = [tok]
                current_label = bio_label[2:]
            elif bio_label.startswith("I-"):
                label_type = bio_label[2:]
                if current_label == label_type and current_entity:
                    # Continuation of current entity
                    current_entity.append(tok)
                else:
                    # FIX: I- tag with no matching B- (orphan) — treat as new B-
                    if current_entity and current_label:
                        entities[current_label].append(" ".join(current_entity))
                    current_entity = [tok]
                    current_label = label_type
            else:
                # "O" label — flush
                if current_entity and current_label:
                    entities[current_label].append(" ".join(current_entity))
                current_entity = []
                current_label = None

        # Flush last entity
        if current_entity and current_label:
            entities[current_label].append(" ".join(current_entity))

        return {k: list(dict.fromkeys(v)) for k, v in entities.items()}

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'clf': self.clf,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'trained': self.trained,
            }, f)
        print(f"  Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "ResumeNERModel":
        m = cls()
        with open(path, 'rb') as f:
            state = pickle.load(f)
        m.clf = state['clf']
        m.label_encoder = state['label_encoder']
        m.feature_names = state['feature_names']
        m.trained = state['trained']
        return m


def train(dataset_path: str, model_save_path: str, eval_save_path: str):
    import time
    print("\n" + "=" * 55)
    print("  RESUME NER MODEL — TRAINING PIPELINE")
    print("=" * 55)

    print("\n[1/5] Loading dataset...")
    with open(dataset_path) as f:
        dataset = json.load(f)
    print(f"  Loaded {len(dataset)} samples")

    train_data = dataset[:1800]
    test_data  = dataset[1800:]

    print("\n[2/5] Collecting feature vocabulary from ALL training samples...")
    t0 = time.time()
    converter = DatasetConverter()
    feature_names = converter.collect_feature_names(train_data)
    print(f"  Feature vocabulary size: {len(feature_names)}")

    print("\n[3/5] Converting to feature matrices...")
    X_train, y_train, _ = converter.convert(train_data, feature_names=feature_names)
    X_test,  y_test,  _ = converter.convert(test_data,  feature_names=feature_names)
    print(f"  Train: {X_train.shape[0]:,} tokens | Test: {X_test.shape[0]:,} tokens")
    print(f"  Features: {X_train.shape[1]} | Classes: {len(set(y_train))}")
    print(f"  Conversion time: {time.time() - t0:.1f}s")

    print("\n[4/5] Training NER model...")
    t0 = time.time()
    model = ResumeNERModel()
    model.feature_names = feature_names
    model.fit(X_train, y_train)
    print(f"  Training time: {time.time() - t0:.1f}s")

    print("\n[5/5] Evaluating on held-out test set...")
    known = set(model.label_encoder.classes_)
    valid_mask = np.array([l in known for l in y_test])
    X_test_v = X_test[valid_mask]
    y_test_v = y_test[valid_mask]

    y_pred_enc = model.clf.predict(X_test_v)
    y_pred = model.label_encoder.inverse_transform(y_pred_enc)
    y_true = y_test_v

    report     = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, zero_division=0)
    overall_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n  Overall Weighted F1: {overall_f1:.4f} ({overall_f1 * 100:.1f}%)")
    print("\n  Per-Entity F1 Scores:")
    entity_f1s = {}
    for label in ENTITY_LABELS:
        if label != "O" and label in report:
            f1 = report[label]['f1-score']
            entity_f1s[label] = f1
            bar = "█" * int(f1 * 20)
            print(f"    {label:<25} F1={f1:.3f}  {bar}")

    os.makedirs(os.path.dirname(os.path.abspath(eval_save_path)), exist_ok=True)
    eval_report = {
        "overall_weighted_f1": float(overall_f1),
        "overall_f1_pct": round(float(overall_f1) * 100, 2),
        "per_entity": entity_f1s,
        "full_report": report,
        "test_samples": len(test_data),
        "train_samples": len(train_data),
        "n_features": X_train.shape[1],
        "n_tokens_train": int(X_train.shape[0]),
        "n_tokens_test": int(X_test_v.shape[0]),
    }
    with open(eval_save_path, 'w') as f:
        json.dump(eval_report, f, indent=2)
    print(f"\n  Evaluation saved → {eval_save_path}")

    model.save(model_save_path)

    print("\n" + "=" * 55)
    print(f"  TRAINING COMPLETE")
    print(f"  Weighted F1:  {overall_f1 * 100:.1f}%")
    print(f"  Model:        {model_save_path}")
    print("=" * 55 + "\n")

    return model, eval_report


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    model, report = train(
        dataset_path=os.path.join(base, "../data/resume_dataset.json"),
        model_save_path=os.path.join(base, "ner_model.pkl"),
        eval_save_path=os.path.join(base, "../evaluation/ner_eval_report.json"),
    )

    print("── Quick Inference Demo ──────────────────────────")
    sample_text = """
Priya Sharma
priya.sharma@gmail.com | +91-9876543210 | linkedin.com/in/priya-sharma

SUMMARY
Senior Data Scientist with 5 years of experience in Python, TensorFlow, and NLP.

EXPERIENCE
Senior Data Scientist | Google | Jan 2020 – Present
• Built BERT-based resume classification model achieving 92% accuracy

EDUCATION
M.Tech in Computer Science | IIT Bombay | 2019

SKILLS
Python, TensorFlow, PyTorch, NLP, SQL, AWS, Docker

CERTIFICATIONS
AWS Certified Machine Learning Specialty
"""
    entities = model.extract_entities(sample_text)
    print("\nExtracted Entities:")
    for label, values in entities.items():
        print(f"  {label}: {values[:3]}")
