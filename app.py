import streamlit as st
import pandas as pd
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GCC AI Hiring Intelligence Platform",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
body { background-color: #f5f7fa; }
[data-testid="stSidebar"] { background-color: #0f172a; }
[data-testid="stSidebar"] * { color: white; }
.comment-box {
    background:#ffffff;
    padding:10px;
    border-radius:8px;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "interview_feedback" not in st.session_state:
    st.session_state.interview_feedback = []

# ---------------- LOAD SKILLS DATABASE ----------------
@st.cache_data
def load_skills_db():
    with open("skills_db.txt", "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

SKILLS_DB = load_skills_db()

# ---------------- RESUME SCREENING HELPERS (UNCHANGED) ----------------
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
        r"over\s*(\d+)\s*years",
        r"(\d+)\s*years of experience",
        r"experience[:\s]+(\d+)\s*years"
    ]
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
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

def rejection_reason(score, experience, missing_skills):
    reasons = []
    if score < 50:
        reasons.append("Low relevance to job description")
    if missing_skills:
        reasons.append(f"Missing skills: {', '.join(missing_skills[:5])}")
    if experience < 2:
        reasons.append("Low experience")
    return "; ".join(reasons)

# ---------------- INTERVIEW & DECISION AI ----------------
def interview_score(comm, tech, culture):
    return round((comm + tech + culture) / 3, 2)

def final_score(resume, interview, assessment):
    return round(resume * 0.5 + interview * 0.3 + assessment * 0.2, 2)

def final_decision(score):
    if score >= 75:
        return "HIRE"
    elif score >= 60:
        return "HOLD"
    else:
        return "REJECT"

# ---------------- ENGAGEMENT AI ----------------
def engagement_score(responses, delays):
    score = 100
    if responses < 2:
        score -= 30
    if delays > 3:
        score -= 40
    return max(score, 0)

def dropoff_risk(score):
    if score < 40:
        return "HIGH"
    elif score < 70:
        return "MEDIUM"
    else:
        return "LOW"

# ---------------- SIDEBAR ----------------
st.sidebar.title("GCC Hiring Platform")
menu = st.sidebar.radio(
    "Menu",
    [
        "Dashboard",
        "Bulk Resume Screening",
        "Interview & Decision",
        "Engagement & Readiness",
        "Interviewee Feedback",
        "Hiring Assistant"
    ]
)

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.title("Executive Hiring Dashboard")
    st.metric("Hiring Intelligence", "End-to-End AI")
    st.metric("Decision Mode", "Explainable")
    st.metric("Candidate Risk", "Predictive")
    st.success("AI-Driven GCC Hiring Intelligence Platform")

# ---------------- MODULE 1: RESUME SCREENING (UNCHANGED UI) ----------------
elif menu == "Bulk Resume Screening":
    st.title("📄 Bulk Resume Screening (PDF Upload)")

    jd = st.text_area("📌 Job Description", height=180)
    uploaded_files = st.file_uploader(
        "📤 Upload Candidate Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Evaluate Candidates"):
        results = []
        jd_skills = extract_jd_skills(jd)

        for file in uploaded_files:
            resume_text = extract_text_from_pdf(file)
            score = compute_match_score(jd.lower(), resume_text)
            decision = hiring_decision(score)
            skills = extract_skills(resume_text)
            exp = extract_experience(resume_text)
            matched, missing = compare_skills(jd_skills, skills)

            results.append({
                "Candidate": file.name.replace(".pdf", ""),
                "Match Score (%)": score,
                "Decision": decision,
                "Experience": exp,
                "Matched Skills": ", ".join(matched),
                "Missing Skills": ", ".join(missing),
                "Rejection Reason": rejection_reason(score, exp, missing) if decision == "REJECT" else ""
            })

        df = pd.DataFrame(results).sort_values("Match Score (%)", ascending=False)
        st.dataframe(df, use_container_width=True)

# ---------------- MODULE 2 ----------------
elif menu == "Interview & Decision":
    st.title("🎤 Interview Evaluation & Decision")

    resume_score = st.slider("Resume Match Score", 0, 100, 70)
    comm = st.slider("Communication", 0, 100, 75)
    tech = st.slider("Technical Skills", 0, 100, 80)
    culture = st.slider("Cultural Fit", 0, 100, 78)
    assessment = st.slider("Assessment Score", 0, 100, 72)

    i_score = interview_score(comm, tech, culture)
    f_score = final_score(resume_score, i_score, assessment)
    decision = final_decision(f_score)

    st.metric("Final Score", f_score)
    st.success(f"Decision: {decision}")

    if st.button("📧 Send Interview Result Email"):
        st.info("Email Sent Successfully (Simulated)")
        st.write(f"Status: {decision} | Final Score: {f_score}%")

# ---------------- MODULE 3 ----------------
elif menu == "Engagement & Readiness":
    st.title("📡 Candidate Engagement & Readiness")

    responses = st.slider("Candidate Responses", 0, 5, 3)
    delays = st.slider("Response Delay (Days)", 0, 7, 2)

    e_score = engagement_score(responses, delays)
    risk = dropoff_risk(e_score)

    st.metric("Engagement Score", e_score)
    st.warning(f"Drop-off Risk: {risk}")

# ---------------- UNIQUE FEATURE ----------------
elif menu == "Interviewee Feedback":
    st.title("💬 Interview Experience (Public Feedback)")

    name = st.text_input("Your Name")
    role = st.text_input("Role Interviewed For")
    rating = st.slider("Rating", 1, 5, 4)
    comment = st.text_area("Your Interview Experience")

    if st.button("Post Feedback"):
        st.session_state.interview_feedback.append({
            "name": name,
            "role": role,
            "rating": rating,
            "comment": comment,
            "time": datetime.now().strftime("%d %b %Y")
        })
        st.success("Feedback posted")

    for fb in reversed(st.session_state.interview_feedback):
        st.markdown(
            f"""
            <div class="comment-box">
            <b>{fb['name']}</b> ({fb['role']}) ⭐{fb['rating']}<br>
            {fb['comment']}<br>
            <small>{fb['time']}</small>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- CHATBOT ----------------
elif menu == "Hiring Assistant":
    st.title("🤖 GCC Hiring Assistant")
    q = st.text_input("Ask something")

    if q:
        if "best" in q.lower():
            st.write("Candidates with high scores and low risk are preferred.")
        elif "drop" in q.lower():
            st.write("Low engagement indicates higher drop-off risk.")
        else:
            st.write("I assist with AI-driven hiring insights.")
