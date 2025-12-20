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

# ---------------- CORE RESUME AI (UNCHANGED) ----------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    return " ".join(
        page.extract_text() or "" for page in reader.pages
    ).lower()

def extract_skills(text):
    return sorted({s for s in SKILLS_DB if s in text})

def extract_jd_skills(jd_text):
    return sorted({s for s in SKILLS_DB if s in jd_text.lower()})

def compare_skills(jd, resume):
    return sorted(set(jd) & set(resume)), sorted(set(jd) - set(resume))

def extract_experience(text):
    years = re.findall(r"(\d+)\s*(?:years|yrs)", text.lower())
    return max(map(int, years)) if years else 0

def compute_match_score(jd, resume):
    tfidf = TfidfVectorizer(stop_words="english")
    vec = tfidf.fit_transform([jd, resume])
    return round(cosine_similarity(vec)[0][1] * 100, 2)

# ---------------- INTERVIEW AI ----------------
def interview_score(c, t, f):
    return round((c + t + f) / 3, 2)

def final_score(resume, interview, assessment):
    return round(resume * 0.5 + interview * 0.3 + assessment * 0.2, 2)

def final_decision(score):
    if score >= 75:
        return "HIRE"
    elif score >= 60:
        return "HOLD"
    return "REJECT"

# ---------------- SIDEBAR ----------------
st.sidebar.title("GCC Hiring Platform")
menu = st.sidebar.radio(
    "Menu",
    [
        "Dashboard",
        "Resume Screening",
        "Interview & Decision",
        "Interviewee Feedback",
        "Hiring Assistant"
    ]
)

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.title("Executive Hiring Dashboard")
    st.success("End-to-End AI-Driven GCC Hiring Intelligence Platform")

# ---------------- RESUME SCREENING ----------------
elif menu == "Resume Screening":
    st.title("📄 Resume Screening")

    jd = st.text_area("Job Description")
    files = st.file_uploader("Upload Resumes", type=["pdf"], accept_multiple_files=True)

    if st.button("Screen"):
        rows = []
        jd_skills = extract_jd_skills(jd)

        for f in files:
            text = extract_text_from_pdf(f)
            score = compute_match_score(jd.lower(), text)
            skills = extract_skills(text)
            matched, missing = compare_skills(jd_skills, skills)

            rows.append({
                "Candidate": f.name.replace(".pdf",""),
                "Score": score,
                "Matched Skills": ", ".join(matched),
                "Missing Skills": ", ".join(missing)
            })

        st.dataframe(pd.DataFrame(rows).sort_values("Score", ascending=False))

# ---------------- INTERVIEW & DECISION ----------------
elif menu == "Interview & Decision":
    st.title("🎤 Interview Evaluation")

    resume_score = st.slider("Resume Score", 0, 100, 70)
    comm = st.slider("Communication", 0, 100, 75)
    tech = st.slider("Technical", 0, 100, 80)
    culture = st.slider("Cultural Fit", 0, 100, 78)
    assessment = st.slider("Assessment Score", 0, 100, 72)

    i_score = interview_score(comm, tech, culture)
    f_score = final_score(resume_score, i_score, assessment)
    decision = final_decision(f_score)

    st.metric("Final Score", f_score)
    st.success(f"Decision: {decision}")

    if st.button("📧 Send Interview Result Email"):
        st.info("Email Sent Successfully (Simulated)")
        st.write(
            f"""
            **Subject:** Interview Result  
            **Status:** {decision}  
            **Final Score:** {f_score}%  
            **Top-10 Shortlisting:** {"Yes" if f_score > 70 else "No"}
            """
        )

# ---------------- INTERVIEWEE FEEDBACK (UNIQUE FEATURE) ----------------
elif menu == "Interviewee Feedback":
    st.title("💬 Interview Experience (Candidate Voice)")

    name = st.text_input("Your Name")
    role = st.text_input("Role Interviewed For")
    rating = st.slider("Interview Experience Rating", 1, 5, 4)
    comment = st.text_area("Share your interview experience")

    if st.button("Post Feedback"):
        st.session_state.interview_feedback.append({
            "name": name,
            "role": role,
            "rating": rating,
            "comment": comment,
            "time": datetime.now().strftime("%d %b %Y")
        })
        st.success("Thank you for your feedback!")

    st.markdown("### 📢 Candidate Comments")
    for fb in reversed(st.session_state.interview_feedback):
        st.markdown(
            f"""
            <div class="comment-box">
            <b>{fb['name']}</b> — {fb['role']} ⭐{fb['rating']}<br>
            {fb['comment']}<br>
            <small>{fb['time']}</small>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- CHATBOT ----------------
elif menu == "Hiring Assistant":
    st.title("🤖 Hiring Assistant")

    q = st.text_input("Ask hiring insights")
    if q:
        if "best" in q.lower():
            st.write("Candidates with high interview and resume scores are recommended.")
        elif "feedback" in q.lower():
            st.write("Interviewee feedback improves transparency and employer branding.")
        else:
            st.write("I assist with AI-driven hiring insights.")
