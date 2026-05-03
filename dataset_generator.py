"""
Synthetic Resume Dataset Generator
Generates 2000+ labeled resume samples with NER annotations.
Covers diverse roles, experience levels, formats, and edge cases.

FIXES:
- Output path now resolves relative to THIS file (not CWD), so it works
  regardless of where the script is invoked from.
- Creates output directory automatically.
- Phone annotation matches the FULL phone string as rendered.
- Degree annotation no longer double-annotates conflicting spans.
- Added __main__ guard so importing the module doesn't run generation.
"""

import random
import re
import json
import os
from typing import List, Dict, Tuple

random.seed(42)

# ── Name components ───────────────────────────────────────────────────────────
FIRST_NAMES = [
    "Aarav","Priya","Rohan","Sneha","Vikram","Ananya","Karan","Divya",
    "Arjun","Pooja","Rahul","Nisha","Amit","Riya","Siddharth","Kavya",
    "James","Sarah","Michael","Emily","David","Jessica","Robert","Ashley",
    "Wei","Mei","Zhang","Lin","Yuki","Kenji","Sakura","Hiroshi",
    "Carlos","Maria","Juan","Sofia","Lucas","Isabella","Pedro","Ana",
    "Mohammed","Fatima","Ali","Zara","Omar","Aisha","Hassan","Layla",
    "Alex","Jordan","Taylor","Morgan","Riley","Casey","Avery","Quinn"
]
LAST_NAMES = [
    "Sharma","Patel","Singh","Kumar","Verma","Gupta","Mehta","Joshi",
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
    "Chen","Wang","Liu","Zhang","Kim","Park","Lee","Tanaka",
    "Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Moore","Taylor",
    "Ahmed","Khan","Ali","Hassan","Ibrahim","Malik","Rahman","Sheikh"
]

EMAIL_DOMAINS = ["gmail.com","yahoo.com","outlook.com","hotmail.com","protonmail.com","icloud.com"]
PHONE_FORMATS = [
    "+91-{}{}{}{}{}{}{}{}{}{}", "+1-{}{}{}-{}{}{}-{}{}{}{}",
    "({}{}{}) {}{}{}-{}{}{}{}", "{}{}{}{}{}{}{}{}{}{}","+44 {}{}{}{}{}{}{}{}{}{}",
]
UNIVERSITIES = [
    "IIT Bombay","IIT Delhi","IIT Madras","NIT Trichy","BITS Pilani","VIT Vellore",
    "Stanford University","MIT","Carnegie Mellon","UC Berkeley","Georgia Tech",
    "University of Michigan","Purdue University","University of Texas Austin",
    "National University of Singapore","IISc Bangalore","Delhi University",
    "Anna University","Manipal University","Amity University"
]
DEGREES = [
    ("B.Tech","Computer Science"),("B.Tech","Information Technology"),
    ("B.E.","Electronics and Communication"),("B.Sc","Computer Science"),
    ("M.Tech","Computer Science"),("M.S.","Data Science"),
    ("M.S.","Computer Science"),("MBA","Business Analytics"),
    ("B.Tech","Electrical Engineering"),("M.Tech","Artificial Intelligence"),
    ("Ph.D","Computer Science"),("B.Sc","Statistics"),("M.Sc","Data Science"),
]
COMPANIES = [
    "Google","Microsoft","Amazon","Meta","Apple","Netflix","Uber","Airbnb",
    "Flipkart","Infosys","TCS","Wipro","HCL","Cognizant","Accenture","IBM",
    "Swiggy","Zomato","Paytm","Razorpay","Freshworks","Zoho","PhonePe",
    "Deloitte","McKinsey","Goldman Sachs","JP Morgan","Citibank","HSBC",
    "DataBricks","Snowflake","Palantir","Stripe","Salesforce","Oracle",
    "Startup Inc","TechCorp","DataLabs","AI Solutions","CloudBase","InnovateTech"
]

ROLES = {
    "data_science": [
        "Data Scientist","Senior Data Scientist","Lead Data Scientist",
        "Machine Learning Engineer","ML Engineer","Senior ML Engineer",
        "Data Science Manager","Principal Data Scientist","Staff Data Scientist"
    ],
    "data_engineering": [
        "Data Engineer","Senior Data Engineer","Lead Data Engineer",
        "Data Pipeline Engineer","Big Data Engineer","Analytics Engineer"
    ],
    "software": [
        "Software Engineer","Senior Software Engineer","Software Developer",
        "Backend Engineer","Full Stack Engineer","Platform Engineer",
        "Site Reliability Engineer","DevOps Engineer","Cloud Engineer"
    ],
    "analytics": [
        "Data Analyst","Senior Data Analyst","Business Analyst",
        "Product Analyst","Marketing Analyst","BI Developer","Analytics Lead"
    ],
    "research": [
        "Research Scientist","Applied Scientist","NLP Engineer",
        "Computer Vision Engineer","AI Researcher","Research Engineer"
    ]
}

SKILL_POOLS = {
    "data_science": [
        "Python","R","TensorFlow","PyTorch","scikit-learn","XGBoost","LightGBM",
        "BERT","GPT","Transformers","NLP","Computer Vision","MLflow","Kubeflow",
        "SQL","Pandas","NumPy","Matplotlib","Seaborn","Plotly","Statistics",
        "A/B Testing","Hypothesis Testing","Feature Engineering","Model Deployment"
    ],
    "data_engineering": [
        "Python","Apache Spark","Kafka","Airflow","Hadoop","dbt","Hive","Flink",
        "PySpark","Databricks","Snowflake","BigQuery","Redshift","AWS Glue",
        "SQL","PostgreSQL","MongoDB","Redis","Terraform","Docker","Kubernetes"
    ],
    "software": [
        "Python","Java","Go","TypeScript","JavaScript","C++","Rust","Kotlin",
        "FastAPI","Flask","Django","Spring Boot","Node.js","React","Vue",
        "Docker","Kubernetes","AWS","GCP","Azure","PostgreSQL","MongoDB","Redis",
        "gRPC","GraphQL","REST API","Microservices","CI/CD","GitHub Actions"
    ],
    "analytics": [
        "SQL","Python","R","Tableau","Power BI","Looker","Excel","Google Sheets",
        "Pandas","NumPy","Statistics","A/B Testing","Google Analytics",
        "BigQuery","Snowflake","dbt","Hypothesis Testing","Data Visualization"
    ],
    "research": [
        "Python","PyTorch","TensorFlow","JAX","Transformers","BERT","GPT","LLM",
        "NLP","Computer Vision","Reinforcement Learning","Statistics","CUDA",
        "NumPy","SciPy","Matplotlib","LaTeX","C++","MATLAB","OpenCV"
    ]
}

CERTIFICATIONS = [
    "AWS Certified Machine Learning Specialty",
    "Google Professional Data Engineer",
    "Google Professional Machine Learning Engineer",
    "Microsoft Azure Data Scientist Associate",
    "AWS Certified Solutions Architect",
    "TensorFlow Developer Certificate",
    "Databricks Certified Associate Developer for Apache Spark",
    "Snowflake SnowPro Core Certification",
    "Certified Kubernetes Administrator (CKA)",
    "AWS Certified Data Analytics Specialty",
    "IBM Data Science Professional Certificate",
    "Deep Learning Specialization (Coursera)",
    "Professional Scrum Master (PSM I)",
    "Project Management Professional (PMP)",
    "Tableau Desktop Specialist",
]

ACHIEVEMENT_TEMPLATES = [
    "Improved model accuracy by {pct}% using {tech}",
    "Reduced inference latency by {pct}% through {tech} optimization",
    "Built {tech}-based pipeline processing {scale}+ daily records",
    "Increased revenue by ${amount}M through ML-driven {feature}",
    "Led team of {n} engineers to deliver {product} serving {users}M+ users",
    "Deployed {tech} model achieving {pct}% F1 score on production data",
    "Reduced customer churn by {pct}% using {tech} model",
    "Achieved {pct}% AUC on {task} using {tech}",
    "Scaled data pipeline to handle {scale}TB+ daily data using {tech}",
    "Reduced cloud costs by ${amount}K/month through {tech} optimization",
    "Automated {process} saving {hours} hours/week",
    "Increased model throughput by {n}x using {tech}",
]


def random_phone() -> str:
    fmt = random.choice(PHONE_FORMATS)
    digits = [str(random.randint(1, 9))] + [str(random.randint(0, 9)) for _ in range(20)]
    try:
        return fmt.format(*digits)
    except (IndexError, KeyError):
        return "+91-9876543210"


def random_achievement(skills: List[str]) -> str:
    tech = random.choice(skills[:8] if len(skills) >= 8 else skills)
    tmpl = random.choice(ACHIEVEMENT_TEMPLATES)
    return tmpl.format(
        pct=random.choice([15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 92, 95]),
        tech=tech,
        scale=random.choice(["1M", "5M", "10M", "50M", "100M", "1B", "2TB", "5TB"]),
        amount=random.choice([1, 2, 3, 5, 10, 15, 20]),
        feature=random.choice(["recommendations", "fraud detection", "personalization", "search ranking"]),
        n=random.choice([2, 3, 4, 5, 6, 8]),
        product=random.choice(["recommendation engine", "fraud detection system", "data platform", "ML pipeline"]),
        users=random.choice([1, 5, 10, 50, 100]),
        task=random.choice(["binary classification", "multi-class classification", "regression"]),
        process=random.choice(["reporting", "data ingestion", "model retraining", "deployment"]),
        hours=random.choice([5, 10, 15, 20, 40]),
    )


def annotate(text: str, substring: str, label: str) -> Tuple[int, int, str]:
    """Return (start, end, label) for the FIRST occurrence of substring in text."""
    idx = text.find(substring)
    if idx == -1:
        raise ValueError(f"Substring not found: {substring!r}")
    return (idx, idx + len(substring), label)


def generate_resume(role_type: str, years_exp: int) -> Dict:
    """Generate a single resume sample with NER annotations."""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"

    first_lower = first.lower().replace(" ", ".")
    last_lower = last.lower()
    suffix = random.choice(["", str(random.randint(1, 99)), str(random.randint(1900, 2005))])
    domain = random.choice(EMAIL_DOMAINS)
    email = f"{first_lower}.{last_lower}{suffix}@{domain}"

    phone = random_phone()

    li_handle = f"{first_lower}-{last_lower}".replace("..", ".")
    linkedin = f"linkedin.com/in/{li_handle}"
    github = f"github.com/{first_lower[0]}{last_lower}"

    roles_list = ROLES[role_type]
    skills_pool = SKILL_POOLS[role_type]
    role = random.choice(roles_list)

    n_skills = random.randint(8, 18)
    chosen_skills = random.sample(skills_pool, min(n_skills, len(skills_pool)))

    degree_tuple = random.choice(DEGREES)
    degree_str = f"{degree_tuple[0]} in {degree_tuple[1]}"
    university = random.choice(UNIVERSITIES)
    grad_year = 2024 - years_exp - random.randint(0, 2)

    n_certs = random.randint(0, 2)
    chosen_certs = random.sample(CERTIFICATIONS, n_certs)

    # Build text
    sep = random.choice([" | ", " • ", " · ", " – ", "\n"])
    contact_line = f"{email}{sep}{phone}{sep}{linkedin}"
    if random.random() > 0.3:
        contact_line += f"{sep}{github}"

    summary_snippets = [
        f"{role} with {years_exp}+ years of experience in {', '.join(chosen_skills[:3])}.",
        f"Proven track record of delivering ML solutions at scale.",
        f"Passionate about {random.choice(['NLP', 'computer vision', 'data engineering', 'MLOps', 'generative AI'])}.",
    ]
    summary = " ".join(summary_snippets[:random.randint(1, 3)])

    # Experience section
    n_jobs = min(years_exp // 2 + 1, 4)
    experience_lines = []
    companies_used = random.sample(COMPANIES, min(n_jobs, len(COMPANIES)))
    for i, company in enumerate(companies_used):
        end_year = 2024 - i * 2
        start_year = end_year - random.randint(1, 3)
        period = f"Jan {start_year} – {'Present' if i == 0 else f'Dec {end_year}'}"
        exp_role = random.choice(roles_list)
        experience_lines.append(f"{exp_role} | {company} | {period}")
        n_bullets = random.randint(2, 4)
        for _ in range(n_bullets):
            achievement = random_achievement(chosen_skills)
            experience_lines.append(f"• {achievement}")

    skills_text = ", ".join(chosen_skills)

    cert_lines = []
    for cert in chosen_certs:
        cert_lines.append(cert)

    parts = [name, contact_line, ""]
    if random.random() > 0.2:
        parts += ["SUMMARY", summary, ""]
    parts += ["EXPERIENCE"] + experience_lines + [""]
    parts += ["EDUCATION", f"{degree_str} | {university} | {grad_year}", ""]
    parts += ["SKILLS", skills_text]
    if cert_lines:
        parts += ["", "CERTIFICATIONS"] + cert_lines

    text = "\n".join(parts)

    # Build annotations
    annotations = []
    try:
        annotations.append(annotate(text, name, "PERSON"))
    except ValueError:
        pass
    try:
        annotations.append(annotate(text, email, "EMAIL"))
    except ValueError:
        pass
    try:
        annotations.append(annotate(text, phone, "PHONE"))
    except ValueError:
        pass
    try:
        annotations.append(annotate(text, linkedin, "URL"))
    except ValueError:
        pass
    try:
        annotations.append(annotate(text, university, "UNIVERSITY"))
    except ValueError:
        pass
    # Annotate degree (just the abbreviation part to avoid span overlap)
    try:
        annotations.append(annotate(text, degree_tuple[0], "DEGREE"))
    except ValueError:
        pass
    # Annotate companies
    for company in companies_used[:2]:
        try:
            annotations.append(annotate(text, company, "COMPANY"))
        except ValueError:
            pass
    # Annotate role
    try:
        annotations.append(annotate(text, role, "JOB_TITLE"))
    except ValueError:
        pass
    # Annotate first 3 skills
    for skill in chosen_skills[:3]:
        try:
            annotations.append(annotate(text, skill, "SKILL"))
        except ValueError:
            pass
    # Annotate first cert
    if chosen_certs:
        try:
            annotations.append(annotate(text, chosen_certs[0], "CERTIFICATION"))
        except ValueError:
            pass

    # Deduplicate overlapping annotations (keep first occurrence)
    seen_spans = set()
    clean_annotations = []
    for ann in annotations:
        span = (ann[0], ann[1])
        if span not in seen_spans and ann[0] != -1:
            seen_spans.add(span)
            clean_annotations.append(ann)

    return {"text": text, "annotations": clean_annotations}


def generate_dataset(n: int = 2000) -> List[Dict]:
    role_types = list(ROLES.keys())
    exp_levels = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    dataset = []
    for i in range(n):
        role_type = role_types[i % len(role_types)]
        years_exp = random.choice(exp_levels)
        try:
            sample = generate_resume(role_type, years_exp)
            dataset.append(sample)
        except Exception as e:
            print(f"[WARN] Skipped sample {i}: {e}")
    return dataset


if __name__ == "__main__":
    # FIX: resolve output path relative to THIS file, not CWD
    _HERE = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(_HERE, "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "resume_dataset.json")

    print(f"Generating 2000 resume samples...")
    dataset = generate_dataset(2000)
    with open(out_path, "w") as f:
        json.dump(dataset, f)
    print(f"Dataset saved → {out_path} ({len(dataset)} samples)")

    # Quick sanity check
    sample = dataset[0]
    print(f"\nSample text preview:\n{sample['text'][:300]}...")
    print(f"\nAnnotations: {sample['annotations'][:4]}")
