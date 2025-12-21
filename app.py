import streamlit as st
import pandas as pd
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GCC AI Hiring Platform",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
h1, h2, h3 { font-weight: 700; }
.rank-badge {
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    color: white;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 14px;
}
.decision-hire { color: #16a34a; font-weight: bold; }
.decision-review { color: #ca8a04; font-weight: bold; }
.decision-reject { color: #dc2626; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------- GROQ CLIENT ----------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
# ---------------- SESSION STATE ----------------
if "interview_scores" not in st.session_state:
    st.session_state["interview_scores"] = {}

if "final_decisions" not in st.session_state:
    st.session_state["final_decisions"] = {}

# ---------------- LOAD SKILLS DATABASE ----------------
@st.cache_data
def load_skills_db():
    with open("skills_db.txt", "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

SKILLS_DB = load_skills_db()

# ---------------- HELPERS (UNCHANGED CORE LOGIC) ----------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = []
    for page in reader.pages:
        if page.extract_text():
            text.append(page.extract_text())
    return " ".join(text).lower()

def extract_skills(text):
    return sorted({s for s in SKILLS_DB if s in text})

def extract_jd_skills(jd_text):
    jd_text = jd_text.lower()
    return sorted({s for s in SKILLS_DB if s in jd_text})

def compare_skills(jd_skills, resume_skills):
    matched = sorted(set(jd_skills) & set(resume_skills))
    missing = sorted(set(jd_skills) - set(resume_skills))
    return matched, missing

def extract_experience(text):
    matches = re.findall(r"(\d+)\+?\s*(years|yrs)", text.lower())
    return max([int(m[0]) for m in matches], default=0)

def compute_match_score(jd, resume):
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform([jd, resume])
    return round(cosine_similarity(vectors)[0][1] * 100, 2)

# ---------------- AI DECISION USING GROQ ----------------
def ai_evaluation(jd, resume, score, matched, missing, experience):
    prompt = f"""
You are an AI hiring expert for a Global Capability Center.

Job Description:
{jd}

Resume:
{resume[:2000]}

Match Score: {score}%
Experience: {experience} years
Matched Skills: {matched}
Missing Skills: {missing}

Rules:
- Strong match → HIRE
- Partial match → REVIEW
- Weak relevance → REJECT

Respond ONLY in this format:
Decision: <HIRE/REVIEW/REJECT>
Reason: <short explanation>
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# ---------------- UI ----------------
st.title("📄 GCC AI Resume Screening Platform")

jd = st.text_area("📌 Job Description", height=180)

uploaded_files = st.file_uploader(
    "📤 Upload Candidate Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------- EVALUATION ----------------
if st.button("🚀 Evaluate Candidates"):
    if not jd or not uploaded_files:
        st.warning("Please provide Job Description and upload resumes.")
    else:
        results = []
        jd_skills = extract_jd_skills(jd)

        for file in uploaded_files:
            resume_text = extract_text_from_pdf(file)
            score = compute_match_score(jd.lower(), resume_text)
            skills = extract_skills(resume_text)
            exp = extract_experience(resume_text)
            matched, missing = compare_skills(jd_skills, skills)

            ai_result = ai_evaluation(jd, resume_text, score, matched, missing, exp)

            decision = (
                "HIRE" if "HIRE" in ai_result else
                "REVIEW" if "REVIEW" in ai_result else
                "REJECT"
            )

            results.append({
                "Candidate": file.name.replace(".pdf", ""),
                "Match Score (%)": score,
                "Decision": decision,
                "Experience (Years)": exp,
                "Matched Skills": ", ".join(matched),
                "Missing Skills": ", ".join(missing),
                "AI Evaluation": ai_result
            })

        df = pd.DataFrame(results).sort_values(
            by="Match Score (%)",
            ascending=False
        ).reset_index(drop=True)

        df["Rank"] = df.index + 1
        st.session_state["screening_df"] = df

        st.success("✅ Screening completed successfully")

# ---------------- RESULTS ----------------
if "screening_df" in st.session_state:
    df = st.session_state["screening_df"]

    st.markdown("## 🧠 AI Screening Results")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ---------------- SHORTLISTING ----------------
    st.markdown("## ⭐ Shortlisting for Interview")

    top_n = st.slider(
        "Select Top N Candidates",
        min_value=1,
        max_value=min(10, len(df)),
        value=5
    )

    shortlisted_df = df.head(top_n)

    st.success(f"Top {top_n} candidates shortlisted")

    st.dataframe(shortlisted_df, use_container_width=True, hide_index=True)

    csv = shortlisted_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Shortlisted CSV",
        data=csv,
        file_name=f"top_{top_n}_candidates.csv",
        mime="text/csv"
    )

    # ---------------- DETAILS ----------------
    st.markdown("## 🔍 Candidate Details")

    for _, row in df.iterrows():
        decision_class = (
            "decision-hire" if row["Decision"] == "HIRE" else
            "decision-review" if row["Decision"] == "REVIEW" else
            "decision-reject"
        )

        with st.expander(f"🏅 Rank {row['Rank']} — {row['Candidate']}"):
            st.markdown(f"""
            <span class="rank-badge">Rank {row['Rank']}</span><br><br>
            <b>Decision:</b> <span class="{decision_class}">{row['Decision']}</span><br>
            <b>Match Score:</b> {row['Match Score (%)']}%<br>
            <b>Experience:</b> {row['Experience (Years)']} years<br>
            <b>Matched Skills:</b> {row['Matched Skills']}<br>
            <b>Missing Skills:</b> {row['Missing Skills']}
            """, unsafe_allow_html=True)

            st.info(row["AI Evaluation"])
            # =====================================================
# 🎤 INTERVIEW SCORING & FINAL DECISION (HUMAN-IN-LOOP)
# =====================================================

st.markdown("---")
st.markdown("## 🎤 Interview Evaluation & Final Decision")

selected_candidate = st.selectbox(
    "Select Candidate for Interview Evaluation",
    df["Candidate"].tolist()
)

st.markdown("### 🧠 Interview Scoring")

col1, col2 = st.columns(2)

with col1:
    technical = st.slider("Technical Skills", 1, 5)
    communication = st.slider("Communication", 1, 5)

with col2:
    problem_solving = st.slider("Problem Solving", 1, 5)
    cultural_fit = st.slider("Cultural Fit", 1, 5)

interview_score = round(
    (technical * 0.4 +
     communication * 0.25 +
     problem_solving * 0.25 +
     cultural_fit * 0.1) * 20,
    2
)

st.metric("Interview Score (/100)", interview_score)

if st.button("✅ Submit Interview Feedback"):
    st.session_state["interview_scores"][selected_candidate] = interview_score
    st.success("Interview feedback recorded")

# ---------------- FINAL TABULATION ----------------
st.markdown("## 📊 Final Hiring Tabulation")

final_rows = []

for _, row in df.iterrows():
    interview = st.session_state["interview_scores"].get(row["Candidate"], 0)
    final_score = round(row["Match Score (%)"] * 0.6 + interview * 0.4, 2)

    final_rows.append({
        "Candidate": row["Candidate"],
        "Resume Score": row["Match Score (%)"],
        "Interview Score": interview,
        "Final Score": final_score
    })

final_df = pd.DataFrame(final_rows).sort_values("Final Score", ascending=False)
final_df["Final Rank"] = range(1, len(final_df) + 1)

st.dataframe(final_df, use_container_width=True, hide_index=True)

# ---------------- HUMAN DECISION ----------------
st.markdown("## 🧑‍⚖️ Human Decision Control")

for _, row in final_df.iterrows():
    with st.expander(f"🏅 Rank {row['Final Rank']} — {row['Candidate']}"):
        st.write(f"**Resume Score:** {row['Resume Score']}%")
        st.write(f"**Interview Score:** {row['Interview Score']}%")
        st.write(f"**Final Score:** {row['Final Score']}%")

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button(f"✅ Hire {row['Candidate']}", key=f"hire_{row['Candidate']}"):
                st.session_state["final_decisions"][row["Candidate"]] = "HIRE"
                st.success("Marked as HIRE")

        with c2:
            if st.button(f"🟡 Review {row['Candidate']}", key=f"review_{row['Candidate']}"):
                st.session_state["final_decisions"][row["Candidate"]] = "REVIEW"
                st.warning("Marked for REVIEW")

        with c3:
            if st.button(f"❌ Reject {row['Candidate']}", key=f"reject_{row['Candidate']}"):
                st.session_state["final_decisions"][row["Candidate"]] = "REJECT"

                rejection_email = f"""
Dear {row['Candidate']},

Thank you for your time and interest in our hiring process.

Evaluation Summary:
- Resume Score: {row['Resume Score']}%
- Interview Score: {row['Interview Score']}%
- Final Score: {row['Final Score']}%

After careful consideration, we will not proceed further.

Regards,
GCC Hiring Team
"""
                st.error("Candidate Rejected")
                st.info("📧 Rejection Email Sent (Simulated)")
                st.code(rejection_email)

            

