import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I'm your AI Hiring Assistant. I can help with resume screening, interview scheduling, candidate insights, and hiring analytics. What would you like to know?"}
    ]

# Mock data
@st.cache_data
def load_mock_data():
    candidates = pd.DataFrame({
        'name': ['Priya Sharma', 'Rahul Kumar', 'Ananya Iyer', 'Arjun Patel', 'Meera Joshi', 
                 'Karthik Nair', 'Sneha Reddy', 'Vikram Singh', 'Pooja Desai', 'Amit Verma'],
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
    # Key Metrics
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
    
    # Visual Funnel
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
    
    # Alerts and Insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚠️ Priority Alerts")
        
        st.markdown("""
        <div class='alert-high'>
            <strong>🔴 High Drop-off Risk</strong><br>
            <span style='font-size: 0.9rem;'>Ananya Iyer - DevOps Engineer (68% engagement)<br>
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
    
    # Top Candidates
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

# AI Screening Page
elif page == "🔍 AI Screening":
    st.markdown("### 🔍 AI Resume Screening")
    
    # Upload Section
    with st.expander("📤 Upload New Resumes", expanded=True):
        uploaded_files = st.file_uploader(
            "Drop resume files here (PDF, DOCX)",
            type=['pdf', 'docx'],
            accept_multiple_files=True
        )
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🚀 Start AI Screening", type="primary", use_container_width=True):
                if uploaded_files:
                    with st.spinner("AI is analyzing resumes..."):
                        import time
                        time.sleep(2)
                    st.success(f"✅ Screened {len(uploaded_files)} resumes!")
                    st.balloons()
    
    st.markdown("---")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        score_filter = st.slider("Minimum Match Score", 0, 100, 70)
    with col2:
        role_filter = st.multiselect("Filter by Role", candidates_df['role'].unique())
    with col3:
        status_filter = st.multiselect("Filter by Status", candidates_df['status'].unique())
    
    # Apply filters
    filtered_df = candidates_df[candidates_df['score'] >= score_filter]
    if role_filter:
        filtered_df = filtered_df[filtered_df['role'].isin(role_filter)]
    if status_filter:
        filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]
    
    st.markdown(f"### 📋 Screened Candidates ({len(filtered_df)} results)")
    
    # Display candidates
    for idx, row in filtered_df.iterrows():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"#### {row['name']}")
            st.caption(f"{row['role']} • {row['experience']}")
            
            skills = row['skills'].split(', ')
            skills_html = ''.join([f"<span class='skill-badge'>{skill}</span>" for skill in skills[:5]])
            st.markdown(skills_html, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<div style='text-align: center;'><div style='font-size: 2rem; font-weight: bold; color: #4F46E5;'>{row['score']}%</div><div style='font-size: 0.85rem; color: #6B7280;'>Match Score</div></div>", unsafe_allow_html=True)
        
        with col3:
            risk_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}
            st.markdown(f"**Risk:** {risk_color[row['risk']]} {row['risk'].upper()}")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("👁️ View", key=f"view_{idx}", use_container_width=True):
                    st.info(f"Profile: {row['name']}")
            with col_b:
                if st.button("✉️ Email", key=f"contact_{idx}", use_container_width=True):
                    st.success(f"Sent to {row['name']}")
        
        st.markdown("---")

# Interviews Page
elif page == "📅 Interviews":
    st.markdown("### 📅 Interview Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🗓️ Upcoming Interviews")
        
        interviews = [
            {"name": "Priya Sharma", "role": "Senior SWE", "time": "Today, 2:00 PM", "interviewer": "Vikram Singh"},
            {"name": "Rahul Kumar", "role": "Data Scientist", "time": "Tomorrow, 10:00 AM", "interviewer": "Sneha Reddy"},
            {"name": "Arjun Patel", "role": "Full Stack", "time": "Dec 23, 3:00 PM", "interviewer": "Amit Verma"},
            {"name": "Meera Joshi", "role": "Backend Dev", "time": "Dec 24, 11:00 AM", "interviewer": "Karthik Nair"}
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
            {"name": "Meera Joshi", "role": "Backend Dev", "date": "Dec 18", "status": "Awaiting Feedback"},
            {"name": "Karthik Nair", "role": "Frontend Dev", "date": "Dec 19", "status": "Partial Feedback"},
            {"name": "Sneha Reddy", "role": "ML Engineer", "date": "Dec 20", "status": "Awaiting Feedback"}
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
    
    # Search and filters
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search = st.text_input("🔍 Search", placeholder="Name, role, or skills...")
    with col2:
        risk_filter = st.selectbox("Risk Level", ["All", "Low", "Medium", "High"])
    with col3:
        sort_by = st.selectbox("Sort by", ["Match Score", "Name", "Status"])
    
    # Display table
    display_df = candidates_df.copy()
    
    # Apply filters
    if search:
        display_df = display_df[
            display_df['name'].str.contains(search, case=False) |
            display_df['role'].str.contains(search, case=False) |
            display_df['skills'].str.contains(search, case=False)
        ]
    
    if risk_filter != "All":
        display_df = display_df[display_df['risk'] == risk_filter.lower()]
    
    # Sort
    if sort_by == "Match Score":
        display_df = display_df.sort_values('score', ascending=False)
    elif sort_by == "Name":
        display_df = display_df.sort_values('name')
    
    # Display as cards
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
    
    # Download
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
    
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['screen', 'resume', 'candidate']):
            response = "📊 Top candidates this week:\n\n• **Priya Sharma** (92%) - Senior SWE\n• **Sneha Reddy** (90%) - ML Engineer\n• **Rahul Kumar** (88%) - Data Scientist\n\nNeed detailed reports?"
        
        elif any(word in prompt_lower for word in ['schedule', 'interview']):
            response = "📅 8 interviews scheduled:\n\n**Today:**\n• Priya Sharma - 2:00 PM\n\n**Tomorrow:**\n• Rahul Kumar - 10:00 AM\n\nWant me to auto-schedule more?"
        
        elif any(word in prompt_lower for word in ['offer', 'drop', 'risk']):
            response = "⚠️ **Alert:**\n\n**Ananya Iyer** - 68% engagement (HIGH risk)\n\n**Actions:**\n1. Expedite decision (48h)\n2. Personalized outreach\n3. Consider +8-12% adjustment"
        
        elif any(word in prompt_lower for word in ['insight', 'analytics']):
            response = "📈 **Insights:**\n\n• Time-to-hire: 18 days (↓15%)\n• Acceptance rate: 88%\n• Python most in-demand\n• 78% prefer hybrid\n• LinkedIn best source"
        
        else:
            response = "I can help with:\n\n🔍 Resume screening\n📅 Interview scheduling\n📊 Analytics\n⚠️ Risk prediction\n\nWhat would you like?"
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>"
    "🧠 GCC Hiring Intelligence Platform | Hackathon 2025"
    "</div>",
    unsafe_allow_html=True
)
