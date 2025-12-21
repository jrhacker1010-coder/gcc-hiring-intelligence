# =====================================================
# GCC AI HIRING PLATFORM – FINAL MERGED STABLE VERSION
# =====================================================

import streamlit as st
import pandas as pd
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="GCC AI Hiring Platform", layout="wide")

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
if "screening_df" not in st.session_state:
    st.session_state.screening_df = None

if "interview_scores" not in st.session_state:
    st.session_state.interview_scores = {}

# ---------------- LOAD SKILLS DB ----------------
@st.cache_data
def load_skills_db():
    with open("skills_db.txt", "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

SKILLS_DB = load_skills_db()

# ---------------- HELPERS ----------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return " ".join([p.extract_text() for p in reader.pages if p.extract_text()]).lower()

def extract_skills(text):
    return sorted({s for s in SKILLS_DB if s in text})

def extract_experience(text):
    matches = re.findall(r"(\d+)\+?\s*(years|yrs)", text)
    return max([int(m[0]) for m in matches], default=0)

def compute_match_score(jd, resume):
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform([jd, resume])
    return round(cosine_similarity(vectors)[0][1] * 100, 2)

def ai_evaluation(jd, resume, score, matched, missing, exp):
    prompt = f"""
You are an AI hiring expert for a Global Capability Center.

Job Description:
{jd}

Resume:
{resume[:1500]}

Match Score: {score}
Experience: {exp}
Matched Skills: {matched}
Missing Skills: {missing}

Respond ONLY in this format:
Decision: HIRE / REVIEW / REJECT
Reason: short explanation
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# ---------------- UI ----------------
st.title("📄 GCC AI Resume Screening Platform")

jd = st.text_area("📌 Job Description", height=180)
files = st.file_uploader("📤 Upload Candidate Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

# =====================================================
# RESUME SCREENING
# =====================================================
if st.button("🚀 Evaluate Candidates"):
    if not jd or not files:
        st.warning("Please provide Job Description and upload resumes.")
    else:
        rows = []

        for f in files:
            resume_text = extract_text_from_pdf(f)
            score = compute_match_score(jd.lower(), resume_text)
            skills = extract_skills(resume_text)
            exp = extract_experience(resume_text)
            missing = list(set(SKILLS_DB) - set(skills))

            ai = ai_evaluation(jd, resume_text, score, skills, missing, exp)
            decision = "HIRE" if "HIRE" in ai else "REVIEW" if "REVIEW" in ai else "REJECT"

            rows.append({
                "Candidate": f.name.replace(".pdf", ""),
                "Match Score (%)": score,
                "Decision": decision,
                "Experience (Years)": exp,
                "Matched Skills": ", ".join(skills),
                "Missing Skills": ", ".join(missing),
                "AI Evaluation": ai
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("Match Score (%)", ascending=False).reset_index(drop=True)
        df["Rank"] = df.index + 1

        st.session_state.screening_df = df
        st.success("✅ Resume Screening Completed")

# =====================================================
# RESULTS + INTERVIEW + FINAL DECISION
# =====================================================
if st.session_state.screening_df is not None:
    df = st.session_state.screening_df

    st.markdown("## 🧠 AI Screening Results")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ---------------- INTERVIEW ----------------
    st.markdown("## 🎤 Interview Evaluation")

    candidate = st.selectbox("Select Candidate", df["Candidate"].tolist())

    c1, c2 = st.columns(2)
    with c1:
        tech = st.slider("Technical Skills", 1, 5)
        comm = st.slider("Communication", 1, 5)
    with c2:
        prob = st.slider("Problem Solving", 1, 5)
        culture = st.slider("Cultural Fit", 1, 5)

    interview_score = round((tech*0.4 + comm*0.25 + prob*0.25 + culture*0.1)*20, 2)
    st.metric("Interview Score", interview_score)

    if st.button("💾 Save Interview Score"):
        st.session_state.interview_scores[candidate] = interview_score
        st.success("Interview score saved")

    # ---------------- FINAL TABULATION ----------------
    st.markdown("## 📊 Final Hiring Dashboard")

    final_rows = []
    for _, r in df.iterrows():
        i = st.session_state.interview_scores.get(r["Candidate"], 0)
        final_rows.append({
            "Candidate": r["Candidate"],
            "Resume Score": r["Match Score (%)"],
            "Interview Score": i,
            "Final Score": round(r["Match Score (%)"]*0.6 + i*0.4, 2)
        })

    final_df = pd.DataFrame(final_rows).sort_values("Final Score", ascending=False)
    final_df["Final Rank"] = range(1, len(final_df)+1)

    st.dataframe(final_df, use_container_width=True, hide_index=True)

    # ---------------- HUMAN DECISION ----------------
    st.markdown("## 🧑‍⚖️ Human Final Decision")

    for _, r in final_df.iterrows():
        with st.expander(f"🏅 Rank {r['Final Rank']} — {r['Candidate']}"):
            st.write(f"Resume Score: {r['Resume Score']}%")
            st.write(f"Interview Score: {r['Interview Score']}%")
            st.write(f"Final Score: {r['Final Score']}%")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.button("✅ Hire", key=f"h_{r['Candidate']}")
            with c2:
                st.button("🟡 Review", key=f"r_{r['Candidate']}")
            with c3:
                if st.button("❌ Reject", key=f"x_{r['Candidate']}"):
                    st.error("📧 Rejection Email Sent (Simulated)")
