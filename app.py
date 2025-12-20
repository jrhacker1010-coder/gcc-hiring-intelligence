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
pre { background-color: #111827; color: #e5e7eb; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------- SKILL DATABASE ----------------
SKILLS_DB = [
    "python","java","sql","aws","azure","gcp","docker","kubernetes",
    "machine learning","deep learning","nlp","data analysis",
    "pandas","numpy","tensorflow","pytorch","spark","hadoop"
]

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

def extract_experience(text):
    matches = re.findall(r"(\\d+)\\+?\\s+years", text)
    return max(map(int, matches)) if matches else 0

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

# ---------------- SIDEBAR ----------------
st.sidebar.title("GCC Hiring Platform")
menu = st.sidebar.radio("Menu", ["Dashboard", "Bulk Resume Screening"])

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.title("Executive Hiring Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Resume Input", "Bulk PDF")
    c2.metric("Evaluation Type", "AI + NLP")
    c3.metric("Decision Output", "Hire / Review / Reject")

    st.success("Enterprise-ready GCC Hiring Intelligence Platform")

# ---------------- BULK SCREENING ----------------
elif menu == "Bulk Resume Screening":
    st.title("📄 Bulk Resume Screening (PDF Upload)")

    jd = st.text_area("📌 Job Description", height=180)

    uploaded_files = st.file_uploader(
        "📤 Upload Candidate Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

    debug = st.checkbox("Show extracted resume text (debug)", False)

    if st.button("Evaluate Candidates"):
        if not jd or not uploaded_files:
            st.warning("Please provide job description and upload resumes.")
        else:
            results = []

            for file in uploaded_files:
                resume_text = extract_text_from_pdf(file)

                if debug:
                    st.markdown(f"#### 📄 Extracted Text — {file.name}")
                    st.text(resume_text[:1500])

                if not resume_text.strip():
                    score = 0
                    decision = "REJECT"
                    skills = []
                    exp = 0
                else:
                    score = compute_match_score(jd.lower(), resume_text)
                    decision = hiring_decision(score)
                    skills = extract_skills(resume_text)
                    exp = extract_experience(resume_text)

                results.append({
                    "Candidate": file.name.replace(".pdf", ""),
                    "Match Score (%)": score,
                    "Decision": decision,
                    "Experience (Years)": exp,
                    "Skills Found": ", ".join(skills) if skills else "Not detected"
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
                    st.write(f"**Skills Identified:** {row['Skills Found']}")
