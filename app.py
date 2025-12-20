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

# ---------------- SESSION STORAGE ----------------
if "interview_data" not in st.session_state:
    st.session_state.interview_data = []

if "community_feedback" not in st.session_state:
    st.session_state.community_feedback = []

# ---------------- LOAD SKILLS DATABASE ----------------
@st.cache_data
def load_skills_db():
    with open("skills_db.txt", "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

SKILLS_DB = load_skills_db()

# ---------------- HELPERS (UNCHANGED) ----------------
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
        r"(\d+)\s*yrs"
    ]
    years = []
    for p in patterns:
        for m in re.findall(p, text.lower()):
            years.append(int(m))
    return max(years) if years else 0

def compute_match_score(jd, resume):
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform([jd, resume])
    return round(cosine_similarity(vectors)[0][1] * 100, 2)

def hiring_decision(score):
    if score >= 70:
        return "HIRE"
    elif score >= 40:
        return "REVIEW"
    else:
        return "REJECT"


def rejection_reason(score, experience, missing_skills):
    reasons = []
    if score < 50: reasons.append("Low relevance to JD")
    if missing_skills: reasons.append("Missing skills")
    if experience < 2: reasons.append("Low experience")
    return "; ".join(reasons)

# ---------------- SIDEBAR ----------------
st.sidebar.title("GCC Hiring Platform")
menu = st.sidebar.radio("Menu", [
    "Dashboard",
    "Bulk Resume Screening",
    "Interview Evaluation",
    "Interviewee Community Feedback",
    "Final Decision & Email"
])

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.title("Executive Hiring Dashboard")
    st.metric("Hiring Intelligence", "AI Enabled")
    st.success("Predictive GCC Hiring Platform")

# ---------------- RESUME SCREENING (UNCHANGED) ----------------
elif menu == "Bulk Resume Screening":
    st.title("📄 Bulk Resume Screening (PDF Upload)")

    jd = st.text_area("📌 Job Description", height=180)

    uploaded_files = st.file_uploader(
        "📤 Upload Candidate Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Evaluate Candidates"):
        if not jd or not uploaded_files:
            st.warning("Please provide job description and upload resumes.")
        else:
            results = []
            jd_skills = extract_jd_skills(jd)

            for file in uploaded_files:
                resume_text = extract_text_from_pdf(file)

                if not resume_text.strip():
                    score = 0
                    decision = "REJECT"
                    skills = []
                    exp = 0
                    matched_skills = []
                    missing_skills = jd_skills
                else:
                    base_score = compute_match_score(jd.lower(), resume_text)
                    skill_bonus = min(len(matched_skills) * 5, 20)
                    score = min(base_score + skill_bonus, 100)
                    decision = hiring_decision(score)
                    skills = extract_skills(resume_text)
                    exp = extract_experience(resume_text)
                    matched_skills, missing_skills = compare_skills(jd_skills, skills)

                results.append({
                    "Candidate": file.name.replace(".pdf", ""),
                    "Match Score (%)": score,
                    "Decision": decision,
                    "Experience (Years)": exp,
                    "JD Required Skills": ", ".join(jd_skills) if jd_skills else "Not detected",
                    "Matched Skills": ", ".join(matched_skills) if matched_skills else "None",
                    "Missing Skills": ", ".join(missing_skills) if missing_skills else "None",
                    "Rejection Reason": rejection_reason(score, exp, missing_skills)
                        if decision == "REJECT" else ""
                })

            
                 df = pd.DataFrame(results).sort_values(
                 by="Match Score (%)", ascending=False
                )

            

            st.markdown("### 🧠 AI Screening Results")
            st.dataframe(df, use_container_width=True)

            st.markdown("### 🔍 Candidate Breakdown")
            for _, row in df.iterrows():
                with st.expander(f"📄 {row['Candidate']} — {row['Decision']}"):
                    st.write(f"**Match Score:** {row['Match Score (%)']}%")
                    st.write(f"**Experience:** {row['Experience (Years)']} years")
                    st.write(f"**JD Required Skills:** {row['JD Required Skills']}")
                    st.write(f"**Matched Skills:** {row['Matched Skills']}")
                    st.write(f"**Missing Skills:** {row['Missing Skills']}")

                    if row["Decision"] == "REJECT":
                        st.error(f"❌ Rejection Reason: {row['Rejection Reason']}")

# ---------------- INTERVIEW EVALUATION ----------------
elif menu == "Interview Evaluation":
    st.title("🎤 Interview Evaluation")

    candidate = st.text_input("Candidate Name")
    communication = st.slider("Communication", 1, 5)
    technical = st.slider("Technical Skills", 1, 5)
    confidence = st.slider("Confidence", 1, 5)

    interview_score = round((communication + technical + confidence) / 3 * 20, 2)

    if st.button("Submit Interview Feedback"):
        st.session_state.interview_data.append({
            "Candidate": candidate,
            "Interview Score": interview_score
        })
        st.success("Interview feedback recorded")

    st.dataframe(pd.DataFrame(st.session_state.interview_data))

# ---------------- COMMUNITY FEEDBACK (UNIQUE FEATURE) ----------------
elif menu == "Interviewee Community Feedback":
    st.title("💬 Interview Experience Wall")

    name = st.text_input("Your Name")
    feedback = st.text_area("Share your interview experience")

    if st.button("Post Feedback"):
        st.session_state.community_feedback.append({
            "Name": name,
            "Feedback": feedback
        })

    for f in st.session_state.community_feedback:
        st.info(f"**{f['Name']}**: {f['Feedback']}")

# ---------------- FINAL DECISION & EMAIL ----------------
elif menu == "Final Decision & Email":
    st.title("📩 Final Decision Engine")

    if "resume_df" in st.session_state:
        resume_df = st.session_state.resume_df
        interview_df = pd.DataFrame(st.session_state.interview_data)

        final_df = resume_df.merge(interview_df, on="Candidate", how="left")
        final_df["Final Score"] = final_df["Resume Score"] * 0.6 + final_df["Interview Score"].fillna(0) * 0.4
        final_df = final_df.sort_values("Final Score", ascending=False)

        st.dataframe(final_df)

        st.success("📧 Result emails sent to candidates (simulated)")
    else:
        st.warning("Please complete resume screening first.")







