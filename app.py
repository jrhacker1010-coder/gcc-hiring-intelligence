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

# ---------------- LOAD SKILLS DATABASE ----------------
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

def compare_skills(jd_skills, resume_skills):
    matched = sorted(set(jd_skills) & set(resume_skills))
    missing = sorted(set(jd_skills) - set(resume_skills))
    return matched, missing

def extract_experience(text):
    patterns = [
        r"(\d+)\+?\s*years",
        r"(\d+)\s*yrs",
        r"(\d+)\s*years of experience"
    ]
    years = []
    for p in patterns:
        matches = re.findall(p, text.lower())
        for m in matches:
            years.append(int(m))
    return max(years) if years else 0

def compute_match_score(jd, resume):
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform([jd, resume])
    return round(cosine_similarity(vectors)[0][1] * 100, 2)

# ---------------- INTERVIEW & ASSESSMENT AI ----------------
def interview_feedback_score(comm, tech, culture):
    return round((comm + tech + culture) / 3, 2)

def final_candidate_score(resume, interview, assessment):
    return round(
        (resume * 0.5) +
        (interview * 0.3) +
        (assessment * 0.2),
        2
    )

def final_decision(score):
    if score >= 75:
        return "HIRE"
    elif score >= 60:
        return "HOLD"
    else:
        return "REJECT"

# ---------------- SIDEBAR ----------------
st.sidebar.title("GCC Hiring Platform")
menu = st.sidebar.radio("Menu", ["Dashboard", "Bulk Resume Screening"])

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.title("Executive Hiring Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Resume Screening", "AI + NLP")
    c2.metric("Interview Evaluation", "Structured Feedback")
    c3.metric("Decision Engine", "Explainable AI")

    st.success("AI-Driven GCC Hiring Intelligence Platform")

# ---------------- BULK SCREENING ----------------
elif menu == "Bulk Resume Screening":
    st.title("📄 End-to-End Candidate Evaluation")

    jd = st.text_area("📌 Job Description", height=180)

    uploaded_files = st.file_uploader(
        "📤 Upload Candidate Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.markdown("## 🗣️ Interview Feedback (Structured)")
    col1, col2, col3 = st.columns(3)

    with col1:
        communication = st.slider("Communication", 0, 100, 70)
    with col2:
        technical = st.slider("Technical Skills", 0, 100, 75)
    with col3:
        culture = st.slider("Cultural Fit", 0, 100, 80)

    st.markdown("## 🧪 Assessment Score")
    assessment = st.slider("Assessment Result", 0, 100, 72)

    if st.button("Evaluate Candidates"):
        if not jd or not uploaded_files:
            st.warning("Please upload Job Description and Resumes.")
        else:
            results = []
            jd_skills = extract_jd_skills(jd)

            interview_score = interview_feedback_score(
                communication, technical, culture
            )

            for file in uploaded_files:
                resume_text = extract_text_from_pdf(file)

                resume_score = compute_match_score(jd.lower(), resume_text)
                final_score = final_candidate_score(
                    resume_score, interview_score, assessment
                )

                decision = final_decision(final_score)

                skills = extract_skills(resume_text)
                matched, missing = compare_skills(jd_skills, skills)
                exp = extract_experience(resume_text)

                results.append({
                    "Candidate": file.name.replace(".pdf", ""),
                    "Resume Score": resume_score,
                    "Interview Score": interview_score,
                    "Assessment Score": assessment,
                    "Final Score": final_score,
                    "Final Decision": decision,
                    "Experience": exp,
                    "Matched Skills": ", ".join(matched),
                    "Missing Skills": ", ".join(missing)
                })

            df = pd.DataFrame(results).sort_values(
                by="Final Score", ascending=False
            )

            st.markdown("### 🧠 AI Final Decision Scorecard")
            st.dataframe(df, use_container_width=True)

            for _, row in df.iterrows():
                with st.expander(f"📄 {row['Candidate']} — {row['Final Decision']}"):
                    st.write(f"Resume Score: {row['Resume Score']}%")
                    st.write(f"Interview Score: {row['Interview Score']}%")
                    st.write(f"Assessment Score: {row['Assessment Score']}%")
                    st.success(f"Final Score: {row['Final Score']}%")
                    st.info(f"Recommendation: {row['Final Decision']}")
