import streamlit as st
import pandas as pd
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GCC AI Hiring Decision Platform",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
body { background-color: #f5f7fa; }
[data-testid="stSidebar"] { background-color: #0f172a; }
[data-testid="stSidebar"] * { color: white; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD SKILLS DATABASE (10,000+) ----------------
@st.cache_data
def load_skills_db():
    with open("skills_db.txt", "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

SKILLS_DB = load_skills_db()

# ---------------- HELPERS ----------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return " ".join(text).lower()

def extract_skills(text):
    return sorted({skill for skill in SKILLS_DB if skill in text})

def extract_jd_skills(jd_text):
    jd_text = jd_text.lower()
    return sorted({skill for skill in SKILLS_DB if skill in jd_text})

def extract_experience(text):
    text = text.lower()
    patterns = [
        r"(\d+)\+?\s*years",
        r"(\d+)\s*yrs",
        r"over\s*(\d+)\s*years",
        r"(\d+)\s*years of experience",
        r"experience[:\s]+(\d+)\s*years"
    ]
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                years.append(int(match))
            except:
                pass
    return max(years) if years else 0

def compute_match_score(jd, resume):
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform([jd, resume])
    return round(cosine_similarity(vectors)[0][1] * 100, 2)

def hiring_decision(score):
    if score >= 75:
        return "HIRE"
    elif score >= 50:
        return "REVIEW"
    else:
        return "REJECT"

def rejection_reason(score, experience, resume_skills, jd_skills):
    reasons = []

    missing_skills = list(set(jd_skills) - set(resume_skills))

    if score < 50:
        reasons.append("Low relevance to the job description")

    if missing_skills:
        reasons.append(
            f"Missing required job skills: {', '.join(missing_skills[:5])}"
        )

    if experience < 2:
        reasons.append("Experience does not meet job expectations")

    return "; ".join(reasons) if reasons else "Profile does not sufficiently match job requirements"

# ---------------- SIDEBAR ----------------
st.sidebar.title("GCC Hiring Platform")
menu = st.sidebar.radio("Menu", ["Dashboard", "Bulk Resume Screening"])

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.title("Executive Hiring Dashboard")

    c1, c2, c3 = st.columns(
