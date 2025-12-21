import streamlit as st
import pandas as pd
import re
from datetime import datetime, timedelta
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# Page config
st.set_page_config(
    page_title="GCC Hiring Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4F46E5;
        margin-bottom: 0.5rem;
    }
    .alert-high {
        background-color: #FEE2E2;
        border-left: 4px solid #EF4444;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-low {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .candidate-card {
        border: 2px solid #E5E7EB;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .skill-badge {
        background-color: #EEF2FF;
        color: #4338CA;
        padding: 0.3rem 0.6rem;
        border-radius: 0.4rem;
        margin-right: 0.4rem;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 0.3rem;
    }
    .progress-bar {
        background-color: #E5E7EB;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .progress-fill {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .rank-badge {
        background: linear-gradient(90deg,#6366f1,#8b5cf6);
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Groq client
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    client = None

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I'm your AI Hiring Assistant. I can help with resume screening, interview scheduling, candidate insights, and hiring analytics. What would you like to know?"}
    ]

if "interview_scores" not in st.session_state:
    st.session_state.interview_scores = {}

if "final_decisions" not in st.session_state:
    st.session_state.final_decisions = {}

if "screening_df" not in st.session_state:
    st.session_state.screening_df = None

# Skills database
@st.cache_data
def load_skills_db():
    skills = [
        "python", "java", "javascript", "typescript", "react", "angular", "vue",
        "node.js", "express", "django", "flask", "spring boot", "aws", "azure",
        "gcp", "docker", "kubernetes", "ci/cd", "jenkins", "git", "sql", "nosql",
        "mongodb", "postgresql", "mysql", "redis", "kafka", "rabbitmq",
        "machine learning", "deep learning", "tensorflow", "pytorch", "nlp",
        "computer vision", "data science", "pandas", "numpy", "scikit-learn",
        "rest api", "graphql", "microservices", "agile", "scrum", "jira"
    ]
    return skills

SKILLS_DB = load_skills_db()

# Resume screening functions
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return " ".join([p.extract_text() for p in reader.pages if p.extract_text()]).lower()

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

def skill_match_score(jd_skills, resume_skills):
    if not jd_skills:
        return 0
    return round((len(set(jd_skills) & set(resume_skills)) / len(jd_skills)) * 100, 2)

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
    return round(text_score * 0.5 + skill_score * 0.3 + exp_score * 0.2, 2)

def ai_evaluation(jd, resume, score, matched, missing, exp):
    if not client:
        return "Decision: REVIEW\nReason: AI evaluation not available"
    
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
    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except:
        return "Decision: REVIEW\nReason: AI evaluation failed"

# Mock data with Tamil Nadu names
@st.cache_data
def load_mock_data():
    candidates = pd.DataFrame({
        'name': ['Karthik Raja', 'Deepa Lakshmi', 'Anitha Devi', 'Rajesh Kumar', 'Priya Sundaram', 
                 'Vijay Raman', 'Divya Balaji', 'Murugan Selvan', 'Kavitha Moorthy', 'Senthil Nathan'],
        'role': ['Senior Software Engineer', 'Data Scientist', 'DevOps Engineer', 'Full Stack Developer',
                 'Backend Developer', 'Frontend Developer', 'ML Engineer', 'Cloud Architect',
                 'Product Manager', 'QA Engineer'],
        'score': [92, 88, 85, 78, 82, 75, 90, 87, 80, 73],
        'status': ['Interview Scheduled', 'Technical Round', 'Offer Extended', 'Resume Screened',
                   'Interview Scheduled', 'Resume Screened', 'Offer Extended', 'Technical Round',
                   'Interview Scheduled', 'Resume Screened'],
        'risk': ['low', 'medium', 'high', 'low', 'medium', 'low', 'low', 'medium', 'high', 'low'],
        'experience': ['5 years', '4 years', '6 years', '3 years', '5 years',
                       '3 years', '6 years', '8 years', '7 years', '2 years'],
        'engagement_score': [95, 85, 68, 88, 78, 92, 87, 75, 65, 90],
        'skills': [
            'Python, AWS, React, Django',
            'Machine Learning, TensorFlow, SQL, Python',
            'Kubernetes, Docker, CI/CD, Jenkins',
            'Node.js, MongoDB, Vue.js, Express',
            'Java, Spring Boot, PostgreSQL',
            'React, TypeScript, Redux, CSS',
            'Deep Learning, PyTorch, NLP',
            'AWS, Azure, Terraform, Microservices',
            'Agile, JIRA, Product Strategy',
            'Selenium, Pytest, API Testing'
        ]
    })
    return candidates

candidates_df = load_mock_data()

# Sidebar
with st.sidebar:
    st.markdown("### 🧠 GCC Hiring Intelligence")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🔍 AI Screening", "📅 Interviews", "👥 Candidates", "💬 AI Assistant"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("**Multi-Tenant Mode**")
    tenant = st.selectbox("Select Team", ["Tech Hiring Team", "Data Science Team", "DevOps Team"])
    
    st.markdown("---")
    st.markdown("**Quick Actions**")
    if st.button("📤 Upload Resumes", use_container_width=True):
        st.success("✅ Resume upload ready!")
    if st.button("📧 Send Bulk Email", use_container_width=True):
        st.success("✅ Email composer opened!")
    if st.button("📈 Generate Report", use_container_width=True):
        st.success("✅ Report generated!")

# Main content
st.markdown("<div class='main-header'>🧠 GCC Hiring Intelligence Platform</div>", unsafe_allow_html=True)
st.markdown("**AI-Powered Recruitment Ecosystem** | *Reimagining GCC Hiring*")
st.markdown("---")

# Dashboard Page
if page == "📊 Dashboard":
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Candidates", "247", "↑ 23")
    with col2:
        st.metric("Screened", "189", "↑ 15")
    with col3:
        st.metric("Interviewed", "45", "↑ 8")
    with col4:
        st.metric("Offers", "12", "↑ 3")
    with col5:
        st.metric("Avg Time", "18 days", "↓ 2")
    with col6:
        st.metric("Drop-off", "12%", "↓ 3%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Hiring Funnel")
        funnel_data = [
            ("Applied", 247),
            ("Screened", 189),
            ("Interviewed", 45),
            ("Offered", 12),
            ("Joined", 10)
        ]
        
        for stage, count in funnel_data:
            percentage = (count / 247) * 100
            st.markdown(f"**{stage}**: {count} candidates")
            st.markdown(f"""
            <div class='progress-bar'>
                <div class='progress-fill' style='width: {percentage}%'>{percentage:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🎯 Status Distribution")
        status_counts = candidates_df['status'].value_counts()
        total = len(candidates_df)
        
        colors = {
            "Interview Scheduled": "#4F46E5",
            "Technical Round": "#7C3AED",
            "Offer Extended": "#10B981",
            "Resume Screened": "#F59E0B"
        }
        
        for status, count in status_counts.items():
            percentage = (count / total) * 100
            color = colors.get(status, "#6B7280")
            
            st.markdown(f"**{status}**: {count} ({percentage:.0f}%)")
            st.markdown(f"""
            <div class='progress-bar'>
                <div class='progress-fill' style='width: {percentage}%; background: {color}'>{count}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚠️ Priority Alerts")
        
        st.markdown("""
        <div class='alert-high'>
            <strong>🔴 High Drop-off Risk</strong><br>
            <span style='font-size: 0.9rem;'>Anitha Devi - DevOps Engineer (68% engagement)<br>
            <em>Recommendation: Expedite decision & personalized outreach</em></span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='alert-medium'>
            <strong>🟡 Interview Pending</strong><br>
            <span style='font-size: 0.9rem;'>8 candidates awaiting schedule confirmation<br>
            <em>Action required within 48 hours</em></span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='alert-low'>
            <strong>🟢 Strong Pipeline</strong><br>
            <span style='font-size: 0.9rem;'>15 high-scoring candidates ready for next round<br>
            <em>Quality score: 85% average</em></span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🧠 AI-Powered Insights")
        
        st.info("**Top Skills in Demand**\n- Python (87 candidates)\n- AWS (65 candidates)\n- React (52 candidates)\n- Machine Learning (48 candidates)")
        
        st.success("**Hiring Efficiency**\n- Time-to-hire improved by 15% this month\n- 78% of candidates prefer hybrid roles\n- Best source: LinkedIn (45% conversion)")
        
        st.warning("**Predictions**\n- Expected 18 offers this month (vs 12 actual)\n- 3 candidates likely to decline offers\n- Recommend salary adjustment for DevOps roles")
    
    st.markdown("---")
    st.markdown("#### 🌟 Top Candidates This Week")
    
    top_candidates = candidates_df.nlargest(3, 'score')
    cols = st.columns(3)
    
    for idx, (_, candidate) in enumerate(top_candidates.iterrows()):
        with cols[idx]:
            st.markdown(f"""
            <div class='candidate-card'>
                <h3 style='color: #4F46E5; margin-bottom: 0.5rem;'>{candidate['name']}</h3>
                <p style='color: #6B7280; margin-bottom: 1rem;'>{candidate['role']}</p>
                <div style='font-size: 2rem; font-weight: bold; color: #4F46E5; margin-bottom: 0.5rem;'>{candidate['score']}%</div>
                <p style='font-size: 0.85rem; color: #6B7280;'>Match Score</p>
            </div>
            """, unsafe_allow_html=True)

# AI Screening Page - Using your original code
elif page == "🔍 AI Screening":
    st.markdown("### 🔍 AI Resume Screening & Evaluation")
    
    jd = st.text_area("📌 Job Description", height=180, placeholder="Enter the job description here...")
    files = st.file_uploader(
        "📤 Upload Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if st.button("🚀 Evaluate Candidates", type="primary"):
        if not jd or not files:
            st.warning("Please provide Job Description and upload resumes")
        else:
            with st.spinner("AI is analyzing resumes..."):
                rows = []
                
                for f in files:
                    text = extract_text_from_pdf(f)
                    skills = extract_skills(text)
                    jd_skills = extract_jd_skills(jd)
                    exp = extract_experience(text)
                    missing = list(set(jd_skills) - set(skills))
                    
                    resume_score = compute_resume_score(jd, text, jd_skills, skills, exp)
                    
                    ai = ai_evaluation(jd, text, resume_score, skills, missing, exp)
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
                
            st.success("✅ Screening completed!")
            st.balloons()
    
    # Display results if available
    if st.session_state.screening_df is not None:
        df = st.session_state.screening_df
        
        st.markdown("---")
        st.markdown("## 🧠 Screening Results")
        st.dataframe(df, hide_index=True, use_container_width=True)
        
        # Interview Evaluation
        st.markdown("---")
        st.markdown("## 🎤 Interview Evaluation")
        
        candidate = st.selectbox("Select Candidate for Interview", df["Candidate"].tolist())
        
        c1, c2 = st.columns(2)
        with c1:
            tech = st.slider("Technical Skills", 1, 5, 3)
            comm = st.slider("Communication", 1, 5, 3)
        with c2:
            prob = st.slider("Problem Solving", 1, 5, 3)
            culture = st.slider("Cultural Fit", 1, 5, 3)
        
        interview_score = round((tech*0.4 + comm*0.25 + prob*0.25 + culture*0.1) * 20, 2)
        st.metric("Interview Score", f"{interview_score}%")
        
        if st.button("💾 Save Interview Score"):
            st.session_state.interview_scores[candidate] = interview_score
            st.success(f"✅ Interview score saved for {candidate}")
        
        # Final Hiring Table
        st.markdown("---")
        st.markdown("## 📊 Final Hiring Decision Table")
        
        final = []
        for _, r in df.iterrows():
            i = st.session_state.interview_scores.get(r["Candidate"], 0)
            final.append({
                "Candidate": r["Candidate"],
                "Resume Score": r["Resume Score (%)"],
                "Interview Score": i,
                "Final Score": round(r["Resume Score (%)"] * 0.5 + i * 0.5, 2),
                "Human Decision": st.session_state.final_decisions.get(r["Candidate"], "PENDING")
            })
        
        final_df = pd.DataFrame(final).sort_values("Final Score", ascending=False)
        final_df["Final Rank"] = range(1, len(final_df) + 1)
        st.dataframe(final_df, hide_index=True, use_container_width=True)
        
        # Human Decision Control
        st.markdown("---")
        st.markdown("## 🧑‍⚖️ Human Decision Control")
        
        for _, r in final_df.iterrows():
            with st.expander(f"🏅 Rank {r['Final Rank']} — {r['Candidate']} (Final Score: {r['Final Score']}%)"):
                st.write(f"**Resume Score:** {r['Resume Score']}%")
                st.write(f"**Interview Score:** {r['Interview Score']}%")
                st.write(f"**Final Score:** {r['Final Score']}%")
                
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    if st.button("✅ Hire", key=f"h_{r['Candidate']}"):
                        st.session_state.final_decisions[r["Candidate"]] = "HIRE"
                        st.success("HIRE decision saved")
                        st.rerun()
                
                with c2:
                    if st.button("🟡 Review", key=f"r_{r['Candidate']}"):
                        st.session_state.final_decisions[r["Candidate"]] = "REVIEW"
                        st.warning("REVIEW decision saved")
                        st.rerun()
                
                with c3:
                    if st.button("❌ Reject", key=f"x_{r['Candidate']}"):
                        st.session_state.final_decisions[r["Candidate"]] = "REJECT"
                        st.error("REJECT decision saved")
                        st.rerun()

# Interviews Page
elif page == "📅 Interviews":
    st.markdown("### 📅 Interview Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🗓️ Upcoming Interviews")
        
        interviews = [
            {"name": "Karthik Raja", "role": "Senior SWE", "time": "Today, 2:00 PM", "interviewer": "Murugan Selvan"},
            {"name": "Deepa Lakshmi", "role": "Data Scientist", "time": "Tomorrow, 10:00 AM", "interviewer": "Divya Balaji"},
            {"name": "Rajesh Kumar", "role": "Full Stack", "time": "Dec 23, 3:00 PM", "interviewer": "Senthil Nathan"},
            {"name": "Priya Sundaram", "role": "Backend Dev", "time": "Dec 24, 11:00 AM", "interviewer": "Vijay Raman"}
        ]
        
        for interview in interviews:
            st.markdown(f"""
            <div style='background-color: #EEF2FF; border-left: 4px solid #4F46E5; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
                <strong style='font-size: 1.1rem;'>{interview['name']}</strong><br>
                <span style='color: #6B7280;'>{interview['role']}</span><br>
                <span style='color: #4F46E5; font-weight: 500;'>⏰ {interview['time']}</span><br>
                <span style='font-size: 0.85rem; color: #6B7280;'>Interviewer: {interview['interviewer']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("🎥 Join", key=f"join_{interview['name']}", use_container_width=True):
                    st.success("Opening call...")
            with col_b:
                if st.button("📅", key=f"reschedule_{interview['name']}", use_container_width=True):
                    st.info("Rescheduling...")
            with col_c:
                if st.button("❌", key=f"cancel_{interview['name']}", use_container_width=True):
                    st.warning("Cancelled")
    
    with col2:
        st.markdown("#### ⏳ Feedback Pending")
        
        pending = [
            {"name": "Priya Sundaram", "role": "Backend Dev", "date": "Dec 18", "status": "Awaiting Feedback"},
            {"name": "Vijay Raman", "role": "Frontend Dev", "date": "Dec 19", "status": "Partial Feedback"},
            {"name": "Divya Balaji", "role": "ML Engineer", "date": "Dec 20", "status": "Awaiting Feedback"}
        ]
        
        for item in pending:
            st.markdown(f"""
            <div style='background-color: #FEF3C7; border-left: 4px solid #F59E0B; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
                <strong style='font-size: 1.1rem;'>{item['name']}</strong><br>
                <span style='color: #6B7280;'>{item['role']}</span><br>
                <span style='color: #D97706; font-weight: 500;'>📅 {item['date']}</span><br>
                <span style='font-size: 0.85rem; color: #92400E;'>{item['status']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📧 Remind", key=f"remind_{item['name']}", use_container_width=True):
                st.success("✅ Reminder sent!")
        
        st.markdown("---")
        st.markdown("#### 🤖 Auto-Schedule")
        if st.button("📅 Schedule 5 Interviews", type="primary", use_container_width=True):
            with st.spinner("Finding slots..."):
                import time
                time.sleep(1.5)
            st.success("✅ 5 interviews scheduled!")

# Candidates Page
elif page == "👥 Candidates":
    st.markdown("### 👥 Candidate Pipeline")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search = st.text_input("🔍 Search", placeholder="Name, role, or skills...")
    with col2:
        risk_filter = st.selectbox("Risk Level", ["All", "Low", "Medium", "High"])
    with col3:
        sort_by = st.selectbox("Sort by", ["Match Score", "Name", "Status"])
    
    display_df = candidates_df.copy()
    
    if search:
        display_df = display_df[
            display_df['name'].str.contains(search, case=False) |
            display_df['role'].str.contains(search, case=False) |
            display_df['skills'].str.contains(search, case=False)
        ]
    
    if risk_filter != "All":
        display_df = display_df[display_df['risk'] == risk_filter.lower()]
    
    if sort_by == "Match Score":
        display_df = display_df.sort_values('score', ascending=False)
    elif sort_by == "Name":
        display_df = display_df.sort_values('name')
    
    for idx, row in display_df.iterrows():
        st.markdown(f"""
        <div class='candidate-card'>
            <div style='display: flex; justify-content: space-between; align-items: start;'>
                <div>
                    <h3 style='color: #1F2937; margin-bottom: 0.3rem;'>{row['name']}</h3>
                    <p style='color: #6B7280; margin-bottom: 0.5rem;'>{row['role']}</p>
                </div>
                <div style='text-align: right;'>
                    <div style='font-size: 1.5rem; font-weight: bold; color: #4F46E5;'>{row['score']}%</div>
                    <div style='font-size: 0.75rem; color: #6B7280;'>Match</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.caption(f"**Status:** {row['status']}")
        with col_b:
            risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}
            st.caption(f"**Risk:** {risk_emoji[row['risk']]} {row['risk'].title()}")
        with col_c:
            st.caption(f"**Engagement:** {row['engagement_score']}%")
        with col_d:
            if st.button("Details", key=f"details_{idx}", use_container_width=True):
                st.info(f"Opening {row['name']}")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        csv = display_df.to_csv(index=False)
        st.download_button("📥 CSV", csv, "candidates.csv", "text/csv", use_container_width=True)
    with col2:
        if st.button("📧 Email", use_container_width=True):
            st.success("✅ Opened!")
    with col3:
        if st.button("📊 Report", use_container_width=True):
            st.success("✅ Generated!")

# AI Assistant Page
else:
    st.markdown("### 💬 AI Hiring Assistant")
    st.caption("Ask about candidates, scheduling, analytics, or insights")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
