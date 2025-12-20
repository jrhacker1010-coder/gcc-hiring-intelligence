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

# ---------- LIVE DASHBOARD STATE (ADD-ON ONLY) ----------
if "live_metrics" not in st.session_state:
    st.session_state.live_metrics = {
        "evaluations": 0,
        "scores": [],
        "hires": 0,
        "high_risk": 0
    }

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

    # ---------- LIVE METRICS (ADD-ON VIEW) ----------
    data = st.session_state.live_metrics

    avg_score = (
        round(sum(data["scores"]) / len(data["scores"]), 2)
        if data["scores"]
        else 0
    )

    st.markdown("### Live Demo Metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluations Done", data["evaluations"])
    c2.metric("Avg Live Match Score", f"{avg_score}%")
    c3.metric("Live Hire Decisions", data["hires"])
    c4.metric("High Drop-Off Alerts", data["high_risk"])

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

    # ---- passive dashboard update (resume screening) ----
    if "score" in locals():
        st.session_state.live_metrics["evaluations"] += 1
        st.session_state.live_metrics["scores"].append(score)

# ---------- MODULE 2 ----------
elif menu == "Interview Decision":
    st.title("Interview Evaluation")

    feedback = st.text_area("Interview Feedback")

    if st.button("Get Decision"):
        if "good" in feedback.lower() or "strong" in feedback.lower():
            st.success("Decision: HIRE")
        else:
            st.error("Decision: REJECT")

    # ---- passive dashboard update (interview decision) ----
    if "feedback" in locals():
        if "good" in feedback.lower() or "strong" in feedback.lower():
            st.session_state.live_metrics["hires"] += 1

# ---------- MODULE 3 ----------
elif menu == "Drop-Off Risk":
    st.title("Candidate Engagement Risk")

    responses = st.slider("Candidate Responses Count", 0, 5, 1)

    if responses < 2:
        st.error("High Drop-Off Risk")
    else:
        st.success("Low Drop-Off Risk")

    # ---- passive dashboard update (drop-off risk) ----
    if "responses" in locals() and responses < 2:
        st.session_state.live_metrics["high_risk"] += 1

import requests
import os

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_HEADERS = {
    "Authorization": f"Bearer {st.secrets['HF_API_TOKEN']}"
}

def ai_chat(prompt):
    payload = {
        "inputs": f"You are an AI hiring assistant for Global Capability Centers. Answer clearly.\n\nUser: {prompt}\nAI:"
    }

    response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return "⚠️ AI service temporarily unavailable"


# ---------- CHATBOT ----------
elif menu == "Chatbot":
    st.title("AI Hiring Assistant")

    query = st.text_input("Ask about hiring, candidates, or GCC strategy")

    if query:
        q = query.lower()

    elif menu == "Chatbot":
    st.title("AI Hiring Assistant")

    user_query = st.text_input("Ask anything about hiring, resumes, GCC strategy")

    if st.button("Ask AI"):
        if user_query:
            with st.spinner("Thinking..."):
                answer = ai_chat(user_query)
                st.write(answer)
   
