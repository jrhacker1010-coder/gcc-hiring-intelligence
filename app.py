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
.stDataFrame { background-color: white; }
</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
SKILLS_DB = [
    "python","java","sql","aws","azure","gcp","docker","kubernetes",
    "machine learning","deep learning","nlp","data analysis",
    "pandas","numpy","tensorflow","pytorch","spark","hadoop"
]

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.lower()

def extract_skills(text):
    return sorted({skill for skill in SKILLS_DB if skill in text})

def extract_experience(text):
    matches = re.findall(r"(\\d+)\\+?\\s+years", text)
    return max(map(int, matches)) if matches else 0

def compute_match_score(jd, resume):
    tfidf = TfidfVectorizer()
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
    c1.metric("Hiring Model", "AI + NLP")
    c2.metric("Resume Input", "PDF (Bulk)")
    c3.metric("Decision Output", "Hire / Review / Reject")

    st.success("Enterprise-ready GCC Hiring Intelligence System")

# ---------------- BULK RESUME SCREENING ----------------
elif menu == "Bulk Resume Screening":
    st.title("📄 Bulk Resume Screening (PDF)")

    jd = st.text_area("📌 Job Description", height=180)

    uploaded_files = st.file_uploader(
        "📤 Upload Candidate Resumes (PDF only)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Evaluate Candidates"):
        if not jd or not uploaded_files:
            st.warning("Please provide job description and upload resumes.")
        else:
            results = []

            for file in uploaded_files:
                resume_text = extract_text_from_pdf(file)

                score = compute_match_score(jd.lower(), resume_text)
                decision = hiring_decision(score)

                results.append({
                    "Candidate": file.name.replace(".pdf", ""),
                    "Match Score (%)": score,
                    "Decision": decision,
                    "Skills Found": ", ".join(extract_skills(resume_text)),
                    "Experience (Years)": extract_experience(resume_text)
                })

            df = pd.DataFrame(results).sort_values(
                by="Match Score (%)", ascending=False
            )

            st.markdown("### 🧠 AI Screening Results")
            st.dataframe(df, use_container_width=True)

            st.markdown("### 🔍 Candidate Details")
            for _, row in df.iterrows():
                with st.expander(f"📄 {row['Candidate']} — {row['Decision']}"):
                    st.write(f"**Match Score:** {row['Match Score (%)']}%")
                    st.write(f"**Experience:** {row['Experience (Years)']} years")
                    st.write(f"**Skills Identified:** {row['Skills Found']}")
