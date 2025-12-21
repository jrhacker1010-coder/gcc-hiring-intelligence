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
    st.session_state.interview_scores = {}

if "final_decisions" not in st.session_state:
    st.session_state.final_decisions = {}

# ---------------- SKILLS DB ----------------
@st.cache_data
def load_skills_db():
    with open("skills_db.txt", "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

SKILLS_DB = load_skills_db()

# ---------------- HELPERS ----------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return " ".join(
        [p.extract_text() for p in reader.pages if p.extract_text()]
    ).lower()

def extract_skills(text):
    return sorted({s for s in SKILLS_DB if s in text})

def extract_jd_skills(text):
    return sorted({s for s in SKILLS_DB if s in text.lower()})

def extract_experience(text):
    matches = re.findall(r"(\d+)\+?\s*(years|yrs)", text.lower())
    return max([int(m[0]) for m in matches], default=0)

def compute_match_score(jd, resume):
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform([jd, resume])
    return round(cosine_similarity(vectors)[0][1] * 100, 2)

# ---------------- SCORE FIX (ONLY ADDITION) ----------------
def skill_match_score(jd_skills, resume_skills):
    if not jd_skills:
        return 0
    return round(
        (len(set(jd_skills) & set(resume_skills)) / len(jd_skills)) * 100,
        2
    )

def experience_score(exp):
    if exp >= 5:
        return 100
    elif exp >= 3:
        return 75
    elif exp >= 1:
        return 50
    else:
        return 25

def compute_resume_score(jd, resume_text, jd_skills, resume_skills, exp):
    text_score = compute_match_score(jd, resume_text)
    skill_score = skill_match_score(jd_skills, resume_skills)
    exp_score = experience_score(exp)

    return round(
        text_score * 0.5 +
        skill_score * 0.3 +
        exp_score * 0.2,
        2
    )

# ---------------- AI ----------------
def ai_evaluation(jd, resume, score, matched, missing, exp):
    prompt = f"""
Job Description:
{jd}

Resume:
{resume[:1500]}

Match Score: {score}
Experience: {exp}
Matched Skills: {matched}
Missing Skills: {missing}

Respond only:
Decision: HIRE / REVIEW / REJECT
Reason: short
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# ---------------- UI ----------------
st.title("📄 GCC AI Resume Screening Platform")

jd = st.text_area("📌 Job Description", height=180)
files = st.file_uploader(
    "📤 Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------- SCREENING ----------------
if st.button("🚀 Evaluate Candidates"):
    if not jd or not files:
        st.warning("Please provide JD and resumes")
    else:
        rows = []

        for f in files:
            text = extract_text_from_pdf(f)

            skills = extract_skills(text)
            jd_skills = extract_jd_skills(jd)
            exp = extract_experience(text)
            missing = list(set(jd_skills) - set(skills))

            resume_score = compute_resume_score(
                jd, text, jd_skills, skills, exp
            )

            ai = ai_evaluation(
                jd, text, resume_score, skills, missing, exp
            )
            decision = (
                "HIRE" if "HIRE" in ai else
                "REVIEW" if "REVIEW" in ai else
                "REJECT"
            )

            rows.append({
                "Candidate": f.name.replace(".pdf", ""),
                "Resume Score (%)": resume_score,
                "AI Decision": decision,
                "Experience": exp,
                "Matched Skills": ", ".join(skills),
                "Missing Skills": ", ".join(missing),
                "AI Reason": ai
            })

        df = (
            pd.DataFrame(rows)
            .sort_values("Resume Score (%)", ascending=False)
            .reset_index(drop=True)
        )
        df["Rank"] = df.index + 1
        st.session_state.screening_df = df
        st.success("✅ Screening completed")

# =====================================================
# SAFE ZONE (df EXISTS)
# =====================================================
if "screening_df" in st.session_state:
    df = st.session_state.screening_df

    st.markdown("## 🧠 Screening Results")
    st.dataframe(df, hide_index=True)

    # ---------------- INTERVIEW ----------------
    st.markdown("## 🎤 Interview Evaluation")

    candidate = st.selectbox(
        "Select Candidate",
        df["Candidate"].tolist()
    )

    c1, c2 = st.columns(2)
    with c1:
        tech = st.slider("Technical", 1, 5)
        comm = st.slider("Communication", 1, 5)
    with c2:
        prob = st.slider("Problem Solving", 1, 5)
        culture = st.slider("Cultural Fit", 1, 5)

    interview_score = round(
        (tech*0.4 + comm*0.25 + prob*0.25 + culture*0.1) * 20,
        2
    )
    st.metric("Interview Score", interview_score)

    if st.button("Save Interview Score"):
        st.session_state.interview_scores[candidate] = interview_score
        st.success("Interview score saved")

    # ---------------- FINAL TABLE ----------------
    st.markdown("## 📊 Final Hiring Table")

    final = []
    for _, r in df.iterrows():
        i = st.session_state.interview_scores.get(
            r["Candidate"], 0
        )
        final.append({
            "Candidate": r["Candidate"],
            "Resume Score": r["Resume Score (%)"],
            "Interview Score": i,
            "Final Score": round(
                r["Resume Score (%)"] * 0.5 + i * 0.5,
                2
            ),
            "Human Decision": st.session_state.final_decisions.get(
                r["Candidate"], "PENDING"
            )
        })

    final_df = (
        pd.DataFrame(final)
        .sort_values("Final Score", ascending=False)
    )
    final_df["Final Rank"] = range(1, len(final_df) + 1)
    st.dataframe(final_df, hide_index=True)

    # ---------------- HUMAN DECISION ----------------
    st.markdown("## 🧑‍⚖️ Human Decision Control")

    for _, r in final_df.iterrows():
        with st.expander(
            f"🏅 Rank {r['Final Rank']} — {r['Candidate']}"
        ):
            st.write(f"Resume Score: {r['Resume Score']}%")
            st.write(f"Interview Score: {r['Interview Score']}%")
            st.write(f"Final Score: {r['Final Score']}%")

            c1, c2, c3 = st.columns(3)

            with c1:
                if st.button(
                    "✅ Hire",
                    key=f"h_{r['Candidate']}"
                ):
                    st.session_state.final_decisions[
                        r["Candidate"]
                    ] = "HIRE"
                    st.success("HIRE saved")

            with c2:
                if st.button(
                    "🟡 Review",
                    key=f"r_{r['Candidate']}"
                ):
                    st.session_state.final_decisions[
                        r["Candidate"]
                    ] = "REVIEW"
                    st.warning("REVIEW saved")

            with c3:
                if st.button(
                    "❌ Reject",
                    key=f"x_{r['Candidate']}"
                ):
                    st.session_state.final_decisions[
                        r["Candidate"]
                    ] = "REJECT"
                    st.error("REJECT saved")
