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
.decision-hire { color: green; font-weight: bold; }
.decision-review { color: orange; font-weight: bold; }
.decision-reject { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------- GROQ CLIENT ----------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---------------- SESSION STATE ----------------
if "screening_df" not in st.session_state:
    st.session_state.screening_df = None

if "interview_scores" not in st.session_state:
    st.session_state.interview_scores = {}

# ---------------- SKILLS DB ----------------
@st.cache_data
def load_skills_db():
    with open("skills_db.txt", "r", encoding="utf-8") as f:
        return [l.strip().lower() for l in f if l.strip()]

SKILLS_DB = load_skills_db()

# ---------------- HELPERS ----------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return " ".join([p.extract_text() for p in reader.pages if p.extract_text()]).lower()

def extract_skills(text):
    return [s for s in SKILLS_DB if s in text]

def extract_experience(text):
    years = re.findall(r"(\d+)\+?\s*(years|yrs)", text)
    return max([int(y[0]) for y in years], default=0)

def compute_match_score(jd, resume):
    tfidf = TfidfVectorizer(stop_words="english")
    vec = tfidf.fit_transform([jd, resume])
    return round(cosine_similarity(vec)[0][1] * 100, 2)

# ---------------- AI DECISION ----------------
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

# ===================== UI =====================
st.title("📄 GCC AI Hiring Platform")

jd = st.text_area("📌 Job Description", height=180)
files = st.file_uploader("📤 Upload Resumes", type=["pdf"], accept_multiple_files=True)

# ===================== SCREENING =====================
if st.button("🚀 Evaluate Candidates"):
    rows = []
    for f in files:
        text = extract_text_from_pdf(f)
        score = compute_match_score(jd, text)
        skills = extract_skills(text)
        exp = extract_experience(text)
        missing = list(set(SKILLS_DB) - set(skills))

        ai = ai_evaluation(jd, text, score, skills, missing, exp)
        decision = "HIRE" if "HIRE" in ai else "REVIEW" if "REVIEW" in ai else "REJECT"

        rows.append({
            "Candidate": f.name.replace(".pdf", ""),
            "Match Score (%)": score,
            "Experience": exp,
            "Decision": decision,
            "Matched Skills": ", ".join(skills),
            "Missing Skills": ", ".join(missing),
            "AI Evaluation": ai
        })

    df = pd.DataFrame(rows).sort_values("Match Score (%)", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    st.session_state.screening_df = df

# ===================== RESULTS =====================
if st.session_state.screening_df is not None:
    df = st.session_state.screening_df
    st.markdown("## 🧠 Screening Results")
    st.dataframe(df, hide_index=True)

    # ===================== INTERVIEW =====================
    st.markdown("## 🎤 Interview Evaluation")

    candidate = st.selectbox("Select Candidate", df["Candidate"].tolist())

    t = st.slider("Technical", 1, 5)
    c = st.slider("Communication", 1, 5)
    p = st.slider("Problem Solving", 1, 5)
    f = st.slider("Cultural Fit", 1, 5)

    interview_score = round((t*0.4 + c*0.25 + p*0.25 + f*0.1) * 20, 2)
    st.metric("Interview Score", interview_score)

    if st.button("Save Interview"):
        st.session_state.interview_scores[candidate] = interview_score

    # ===================== FINAL TABULATION =====================
    st.markdown("## 📊 Final Hiring Dashboard")

    final = []
    for _, r in df.iterrows():
        i = st.session_state.interview_scores.get(r["Candidate"], 0)
        final.append({
            "Candidate": r["Candidate"],
            "Resume Score": r["Match Score (%)"],
            "Interview Score": i,
            "Final Score": round(r["Match Score (%)"]*0.6 + i*0.4, 2),
            "AI Reason": r["AI Evaluation"]
        })

    final_df = pd.DataFrame(final).sort_values("Final Score", ascending=False)
    final_df["Final Rank"] = range(1, len(final_df)+1)

    st.dataframe(final_df, hide_index=True)

    # ===================== HUMAN DECISION =====================
    st.markdown("## 🧑‍⚖️ Human Decision")

    for _, r in final_df.iterrows():
        with st.expander(f"🏅 Rank {r['Final Rank']} — {r['Candidate']}"):
            st.write("🧠 **AI Reason:**")
            st.info(r["AI Reason"])
            if st.button("❌ Reject", key=r["Candidate"]):
                st.error("📧 Rejection Email Sent (Simulated)")
