import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
[data-testid="stSidebar"] {
    background-color: #0f172a;
}
[data-testid="stSidebar"] * {
    color: white;
}
</style>
""", unsafe_allow_html=True)


st.set_page_config(
    page_title="GCC Hiring Intelligence Platform",
    layout="wide"
)

# ---------- AI LOGIC ----------
def match_score(jd, resume):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd, resume])
    score = cosine_similarity(vectors)[0][1]
    return round(score * 100, 2)

# ---------- SIDEBAR ----------
st.sidebar.title("GCC Hiring Platform")
menu = st.sidebar.radio(
    "Menu",
    ["Dashboard", "Resume Screening", "Interview Decision", "Drop-Off Risk", "Chatbot"]
)

# ---------- DASHBOARD ----------
if menu == "Dashboard":
    st.title("Executive Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Open Positions", "10")
    col2.metric("Active Candidates", "65")
    col3.metric("Avg Match Score", "82%")

    st.success("AI-powered hiring intelligence for Global Capability Centers")

# ---------- MODULE 1 ----------
elif menu == "Resume Screening":
    st.title("Resume Screening")

    jd = st.text_area("Job Description")
    resume = st.text_area("Candidate Resume")

    if st.button("Evaluate"):
        if jd and resume:
            score = match_score(jd, resume)
            st.metric("Match Score", f"{score}%")
        else:
            st.warning("Please enter both fields")

# ---------- MODULE 2 ----------
elif menu == "Interview Decision":
    st.title("Interview Evaluation")

    feedback = st.text_area("Interview Feedback")

    if st.button("Get Decision"):
        if "good" in feedback.lower() or "strong" in feedback.lower():
            st.success("Decision: HIRE")
        else:
            st.error("Decision: REJECT")

# ---------- MODULE 3 ----------
elif menu == "Drop-Off Risk":
    st.title("Candidate Engagement Risk")

    responses = st.slider("Candidate Responses Count", 0, 5, 1)

    if responses < 2:
        st.error("High Drop-Off Risk")
    else:
        st.success("Low Drop-Off Risk")

# ---------- CHATBOT ----------
elif menu == "Chatbot":
    st.title("AI Hiring Assistant")

    query = st.text_input("Ask about hiring, candidates, or GCC strategy")

    if query:
        q = query.lower()

        if "hire" in q:
            st.write("Hiring decisions are based on AI match score, interview feedback, and engagement risk.")
        elif "resume" in q:
            st.write("Resumes are evaluated using NLP similarity with job descriptions.")
        elif "drop" in q:
            st.write("Low engagement indicates a high candidate drop-off risk.")
        elif "gcc" in q:
            st.write("This platform is tailored for Global Capability Center hiring workflows.")
        else:
            st.write("Please ask a GCC hiring-related question.")


