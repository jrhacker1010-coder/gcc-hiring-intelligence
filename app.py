import streamlit as st
import pandas as pd
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="GCC AI Hiring Platform", layout="wide")

# ---------------- GROQ CLIENT ----------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---------------- LOAD SKILLS DATABASE ----------------
@st.cache_data
def load_skills_db():
    with open("skills_db.txt", "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

SKILLS_DB = load_skills_db()

# ---------------- HELPERS ----------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    return " ".join([p.extract_text() or "" for p in reader.pages]).lower()

def extract_skills(text):
    return sorted({s for s in SKILLS_DB if s in text})

def extract_jd_skills(jd_text):
    jd_text = jd_text.lower()
    return sorted({s for s in SKILLS_DB if s in jd_text})

def compare_skills(jd_skills, resume_skills):
    return list(set(jd_skills) & set(resume_skills)), list(set(jd_skills) - set(resume_skills))

def extract_experience(text):
    matches = re.findall(r"(\d+)\+?\s*(years|yrs)", text.lower())
    return max([int(m[0]) for m in matches], default=0)

def compute_match_score(jd, resume):
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform([jd, resume])
    return round(cosine_similarity(vectors)[0][1] * 100, 2)

def ai_evaluation(jd, resume, score, matched, missing, experience):
    prompt = f"""
Job Description:
{jd}

Resume:
{resume[:2000]}

Match Score: {score}%
Experience: {experience} years
Matched Skills: {matched}
Missing Skills: {missing}

Respond ONLY:
Decision: HIRE / REVIEW / REJECT
Reason: short explanation
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# ---------------- SIDEBAR ----------------
st.sidebar.title("GCC Hiring Platform")
menu = st.sidebar.radio("Menu", ["Bulk Resume Screening"])

# ---------------- BULK SCREENING ----------------
if menu == "Bulk Resume Screening":
    st.title("📄 Bulk Resume Screening with AI")

    jd = st.text_area("📌 Job Description", height=180)
    uploaded_files = st.file_uploader(
        "📤 Upload Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Evaluate Candidates"):
        if not jd or not uploaded_files:
            st.warning("Please provide JD and resumes")
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
                decision = "HIRE" if "HIRE" in ai_result else "REVIEW" if "REVIEW" in ai_result else "REJECT"

                results.append({
                    "Candidate": file.name.replace(".pdf", ""),
                    "Match Score (%)": score,
                    "Decision": decision,
                    "Experience (Years)": exp,
                    "Matched Skills": ", ".join(matched),
                    "Missing Skills": ", ".join(missing),
                    "AI Evaluation": ai_result
                })

            df = pd.DataFrame(results).sort_values("Match Score (%)", ascending=False).reset_index(drop=True)
            df["Rank"] = df.index + 1

            st.session_state["screening_df"] = df

            st.success("✅ Screening completed")
            st.dataframe(df, use_container_width=True)

# ---------------- SHORTLISTING ----------------
if "screening_df" in st.session_state:
    df = st.session_state["screening_df"]

    st.markdown("## ⭐ Shortlisting for Interview")
    top_n = st.slider("Select Top N Candidates", 1, len(df), 5)

    shortlisted_df = df.head(top_n)
    st.dataframe(shortlisted_df, use_container_width=True)

    csv = shortlisted_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Shortlisted CSV",
        csv,
        f"shortlisted_top_{top_n}.csv",
        "text/csv"
    )

    st.markdown("## 🔍 Candidate Details")
    for _, row in df.iterrows():
        with st.expander(f"Rank {row['Rank']} — {row['Candidate']} ({row['Decision']})"):
            st.write(row["AI Evaluation"])
