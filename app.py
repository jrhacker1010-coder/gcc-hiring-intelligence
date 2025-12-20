import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# ---------------- SESSION STATE (LIVE DASHBOARD) ----------------
if "live_metrics" not in st.session_state:
    st.session_state.live_metrics = {
        "evaluations": 0,
        "scores": [],
        "hires": 0,
        "high_risk": 0
    }

# ---------------- RESUME MATCH LOGIC ----------------
def match_score(jd, resume):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd, resume])
    score = cosine_similarity(vectors)[0][1]
    return round(score * 100, 2)

# ---------------- HUGGING FACE AI CHATBOT (FREE) ----------------
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
HF_HEADERS = {
    "Authorization": f"Bearer {st.secrets['HF_API_TOKEN']}"
}

def ai_chat(prompt):
    payload = {
        "inputs": f"You are an AI hiring assistant for Global Capability Centers.\nQuestion: {prompt}"
    }
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return "⚠️ AI service temporarily unavailable"

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

    data = st.session_state.live_metrics
    avg_score = round(sum(data["scores"]) / len(data["scores"]), 2) if data["scores"] else 0

    st.markdown("### Live Demo Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluations Done", data["evaluations"])
    c2.metric("Avg Live Match Score", f"{avg_score}%")
    c3.metric("Hire Decisions", data["hires"])
    c4.metric("High Drop-Off Alerts", data["high_risk"])

# ---------------- RESUME SCREENING ----------------
elif menu == "Resume Screening":
    st.title("Resume Screening")

    jd = st.text_area("Job Description", height=150)
    resume = st.text_area("Candidate Resume", height=200)

    if st.button("Evaluate"):
        if jd and resume:
            score = match_score(jd, resume)
            st.metric("Match Score", f"{score}%")

            st.session_state.live_metrics["evaluations"] += 1
            st.session_state.live_metrics["scores"].append(score)
        else:
            st.warning("Please enter both Job Description and Resume")

# ---------------- INTERVIEW DECISION ----------------
elif menu == "Interview Decision":
    st.title("Interview Evaluation")

    feedback = st.text_area("Interview Feedback")

    if st.button("Get Decision"):
        if "good" in feedback.lower() or "strong" in feedback.lower():
            st.success("Decision: HIRE")
            st.session_state.live_metrics["hires"] += 1
        else:
            st.error("Decision: REJECT")

# ---------------- DROP-OFF RISK ----------------
elif menu == "Drop-Off Risk":
    st.title("Candidate Engagement Risk")

    responses = st.slider("Candidate Responses Count", 0, 5, 1)

    if responses < 2:
        st.error("High Drop-Off Risk")
        st.session_state.live_metrics["high_risk"] += 1
    else:
        st.success("Low Drop-Off Risk")

# ---------------- CHATBOT ----------------
elif menu == "Chatbot":
    st.title("AI Hiring Assistant")

    question = st.text_input("Ask anything about hiring, resumes, or GCC strategy")

    if st.button("Ask AI"):
        if question:
            with st.spinner("Thinking..."):
                answer = ai_chat(question)
                st.write(answer)
