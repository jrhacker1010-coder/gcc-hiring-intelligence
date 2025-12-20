import streamlit as st
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GCC Hiring & Interview Decision Assistant",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
body { background-color: #f5f7fa; }
[data-testid="stSidebar"] { background-color: #0f172a; }
[data-testid="stSidebar"] * { color: white; }
pre { background-color: #0f172a; color: #e5e7eb; padding: 15px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
SKILL_LIBRARY = [
    "python","java","sql","aws","azure","gcp","docker","kubernetes",
    "machine learning","deep learning","nlp","data analysis","pandas",
    "numpy","tensorflow","pytorch","react","node","spark","hadoop"
]

def extract_skills(text):
    text = text.lower()
    return sorted(list({skill for skill in SKILL_LIBRARY if skill in text}))

def extract_experience(text):
    match = re.findall(r"(\d+)\+?\s+years", text.lower())
    return max(map(int, match)) if match else 0

def extract_education(text):
    edu = []
    for degree in ["bachelor","master","phd","b.tech","m.tech","mba"]:
        if degree in text.lower():
            edu.append(degree.upper())
    return edu

def extract_roles(text):
    lines = text.split("\n")
    return [line.strip() for line in lines if "engineer" in line.lower() or "developer" in line.lower()]

def compute_match_score(jd, resume):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd, resume])
    return round(cosine_similarity(vectors)[0][1] * 100, 2)

# ---------------- SIDEBAR ----------------
st.sidebar.title("GCC Hiring Platform")
menu = st.sidebar.radio(
    "Menu",
    ["Dashboard", "AI Resume & JD Evaluation"]
)

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.title("Executive Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Use Case", "GCC Hiring")
    col2.metric("AI Mode", "Offline NLP")
    col3.metric("Output Format", "JSON Only")

    st.success("AI Hiring & Interview Decision Assistant ready for evaluation")

# ---------------- MAIN EVALUATION ----------------
elif menu == "AI Resume & JD Evaluation":
    st.title("AI Hiring & Interview Decision Assistant")

    jd = st.text_area("📌 Job Description", height=200)
    resume = st.text_area("📄 Candidate Resume (Extracted Text)", height=250)

    if st.button("Evaluate Candidate"):
        if not jd or not resume:
            st.warning("Please provide both Job Description and Resume.")
        else:
            # ----- Resume Extraction -----
            resume_skills = extract_skills(resume)
            resume_exp = extract_experience(resume)
            resume_edu = extract_education(resume)
            resume_roles = extract_roles(resume)

            # ----- JD Extraction -----
            jd_skills = extract_skills(jd)
            jd_exp = extract_experience(jd)

            # ----- Comparison -----
            matching_skills = list(set(resume_skills) & set(jd_skills))
            missing_skills = list(set(jd_skills) - set(resume_skills))

            score = compute_match_score(jd, resume)

            if score >= 75:
                decision = "Strong Match"
                recommendation = "Candidate meets most technical and experience requirements."
            elif score >= 50:
                decision = "Partial Match"
                recommendation = "Candidate shows potential but lacks some key requirements."
            else:
                decision = "Not a Match"
                recommendation = "Candidate does not sufficiently match role expectations."

            experience_summary = (
                f"Candidate has approximately {resume_exp} years of experience. "
                f"Minimum required is {jd_exp} years."
            )

            output = {
                "match_score": score,
                "decision": decision,
                "matching_skills": matching_skills,
                "missing_skills": missing_skills,
                "experience_summary": experience_summary,
                "final_recommendation": recommendation
            }

            st.markdown("### 🔍 AI Evaluation Output (JSON)")
            st.code(json.dumps(output, indent=2), language="json")
