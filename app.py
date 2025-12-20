import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GCC Hiring Intelligence Platform",
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

# ---------------- AI MATCH LOGIC ----------------
def match_score(jd, resume):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd, resume])
    score = cosine_similarity(vectors)[0][1]
    return round(score * 100, 2)

# ---------------- SEMANTIC CHATBOT (OFFLINE AI) ----------------
model = SentenceTransformer("all-MiniLM-L6-v2")

INTENTS = {
    "resume screening": "Resume screening uses NLP to compare resumes with job descriptions and calculate relevance scores.",
    "interview decision": "Interview decisions combine interviewer feedback, skill alignment, and hiring benchmarks.",
    "drop-off risk": "Drop-off risk identifies candidates likely to disengage based on communication behavior.",
    "gcc hiring": "GCC hiring focuses on scalable talent acquisition, operational efficiency, and domain expertise.",
    "ai hiring": "AI improves hiring by automating screening, scoring candidates, and predicting hiring outcomes."
}

intent_keys = list(INTENTS.keys())
intent_embeddings = model.encode(intent_keys)

def semantic_chat(query):
    query_embedding = model.encode(query)
    similarities = np.dot(intent_embeddings, query_embedding)
    best_index = np.argmax(similarities)
    return INTENTS[intent_keys[best_index]]

# ---------------- SIDEBAR ----------------
st.sidebar.title("GCC Hiring Platform")
menu = st.sidebar.radio(
    "Menu",
    ["Dashboard", "Resume Screening", "Interview Decision", "Drop-Off Risk", "Chatbot"]
)

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.title("Executive Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Open Positions", "10")
    col2.metric("Active Candidates", "65")
    col3.metric("Avg Match Score", "82%")

    st.success("AI-powered hiring intelligence for Global Capability Centers")

# ---------------- RESUME SCREENING ----------------
elif menu == "Resume Screening":
    st.title("Resume Screening")

    jd = st.text_area("Job Description", height=150)
    resume = st.text_area("Candidate Resume", height=200)

    if st.button("Evaluate"):
        if jd and resume:
            score = match_score(jd, resume)
            st.metric("Match Score", f"{score}%")
        else:
            st.warning("Please enter both Job Description and Resume")

# ---------------- INTERVIEW DECISION ----------------
elif menu == "Interview Decision":
    st.title("Interview Evaluation")

    feedback = st.text_area("Interview Feedback")

    if st.button("Get Decision"):
        if "good" in feedback.lower() or "strong" in feedback.lower():
            st.success("Decision: HIRE")
        else:
            st.error("Decision: REJECT")

# ---------------- DROP-OFF RISK ----------------
elif menu == "Drop-Off Risk":
    st.title("Candidate Engagement Risk")

    responses = st.slider("Candidate Responses Count", 0, 5, 1)

    if responses < 2:
        st.error("High Drop-Off Risk")
    else:
        st.success("Low Drop-Off Risk")

# ---------------- CHATBOT (SEMANTIC AI) ----------------
elif menu == "Chatbot":
    st.title("AI Hiring Assistant")

    user_query = st.text_input("Ask anything about hiring, resumes, or GCC strategy")

    if st.button("Ask"):
        if user_query:
            answer = semantic_chat(user_query)
            st.write(answer)
