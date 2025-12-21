import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I'm your AI Hiring Assistant. I can help with resume screening, interview scheduling, candidate insights, and hiring analytics. What would you like to know?"}
    ]

# Mock data
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
        st.info("Resume upload feature")
    if st.button("📧 Send Bulk Email", use_container_width=True):
        st.info("Email feature")
    if st.button("📈 Generate Report", use_container_width=True):
        st.info("Report generation")

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
        st.metric("Avg Time to Hire", "18 days", "↓ 2 days")
    with col6:
        st.metric("Drop-off Rate", "12%", "↓ 3%")
    
    st.markdown("---")
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Hiring Funnel")
        funnel_data = pd.DataFrame({
            'Stage': ['Applied', 'Screened', 'Interviewed', 'Offered', 'Joined'],
            'Count': [247, 189, 45, 12, 10]
        })
        fig = px.funnel(funnel_data, x='Count', y='Stage', color='Stage')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Candidate Status Distribution")
        status_counts = candidates_df['status'].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index, hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
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
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🚀 Start AI Screening", type="primary", use_container_width=True):
                if uploaded_files:
                    with st.spinner("AI is analyzing resumes..."):
                        import time
                        time.sleep(2)
                    st.success(f"✅ Screened {len(uploaded_files)} resumes successfully!")
    
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
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{row['name']}**")
                st.caption(f"{row['role']} • {row['experience']}")
                
                # Skills badges
                skills = row['skills'].split(', ')
                skills_html = ' '.join([f"<span style='background-color: #E0E7FF; color: #4338CA; padding: 0.2rem 0.5rem; border-radius: 0.3rem; margin-right: 0.3rem; font-size: 0.8rem;'>{skill}</span>" for skill in skills[:4]])
                st.markdown(skills_html, unsafe_allow_html=True)
            
            with col2:
                st.metric("Match Score", f"{row['score']}%")
                st.caption(f"Status: {row['status']}")
            
            with col3:
                risk_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}
                st.markdown(f"**Risk:** {risk_color[row['risk']]} {row['risk'].upper()}")
                if st.button("View Full Profile", key=f"view_{idx}", use_container_width=True):
                    st.info(f"Opening profile for {row['name']}")
        
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
            with st.container():
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
                    st.button("Join Call", key=f"join_{interview['name']}", use_container_width=True)
                with col_b:
                    st.button("Reschedule", key=f"reschedule_{interview['name']}", use_container_width=True)
                with col_c:
                    st.button("Cancel", key=f"cancel_{interview['name']}", use_container_width=True)
    
    with col2:
        st.markdown("#### ⏳ Feedback Pending")
        
        pending = [
            {"name": "Meera Joshi", "role": "Backend Dev", "date": "Dec 18", "status": "Awaiting Feedback"},
            {"name": "Karthik Nair", "role": "Frontend Dev", "date": "Dec 19", "status": "Partial Feedback"},
            {"name": "Sneha Reddy", "role": "ML Engineer", "date": "Dec 20", "status": "Awaiting Feedback"}
        ]
        
        for item in pending:
            with st.container():
                st.markdown(f"""
                <div style='background-color: #FEF3C7; border-left: 4px solid #F59E0B; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
                    <strong style='font-size: 1.1rem;'>{item['name']}</strong><br>
                    <span style='color: #6B7280;'>{item['role']}</span><br>
                    <span style='color: #D97706; font-weight: 500;'>📅 Interviewed: {item['date']}</span><br>
                    <span style='font-size: 0.85rem; color: #92400E;'>{item['status']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Send Reminder", key=f"remind_{item['name']}", use_container_width=True):
                    st.success(f"Reminder sent to interviewer!")
        
        st.markdown("---")
        st.markdown("#### 🤖 Auto-Schedule Next Batch")
        if st.button("Schedule 5 Interviews", type="primary", use_container_width=True):
            with st.spinner("AI is finding optimal time slots..."):
                import time
                time.sleep(1.5)
            st.success("✅ Scheduled 5 interviews for next week!")

# Candidates Page
elif page == "👥 Candidates":
    st.markdown("### 👥 Candidate Pipeline")
    
    # Search and filters
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search = st.text_input("🔍 Search candidates", placeholder="Name, role, or skills...")
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
    
    # Style the dataframe
    def color_risk(val):
        colors = {'low': '#D1FAE5', 'medium': '#FEF3C7', 'high': '#FEE2E2'}
        return f'background-color: {colors.get(val, "white")}'
    
    styled_df = display_df[['name', 'role', 'score', 'status', 'risk', 'engagement_score']].style.applymap(
        color_risk, subset=['risk']
    )
    
    st.dataframe(
        styled_df,
        column_config={
            "name": "Candidate Name",
            "role": "Role",
            "score": st.column_config.ProgressColumn(
                "Match Score",
                format="%d%%",
                min_value=0,
                max_value=100,
            ),
            "status": "Current Status",
            "risk": "Drop-off Risk",
            "engagement_score": st.column_config.ProgressColumn(
                "Engagement",
                format="%d%%",
                min_value=0,
                max_value=100,
            ),
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Download options
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        csv = display_df.to_csv(index=False)
        st.download_button(
            "📥 Download as CSV",
            csv,
            "candidates.csv",
            "text/csv",
            use_container_width=True
        )
    with col2:
        if st.button("📧 Email Selected", use_container_width=True):
            st.info("Bulk email feature")
    with col3:
        if st.button("📊 Generate Report", use_container_width=True):
            st.info("Report generation")

# AI Assistant Page
else:  # AI Assistant
    st.markdown("### 💬 AI Hiring Assistant")
    st.caption("Ask me about candidates, scheduling, analytics, or hiring insights")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about hiring..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Generate response
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['screen', 'resume', 'candidate']):
            response = "📊 I've analyzed 247 resumes this week. Top candidates:\n\n" \
                      "• **Priya Sharma** (92% match) - Senior Software Engineer\n" \
                      "• **Sneha Reddy** (90% match) - ML Engineer\n" \
                      "• **Rahul Kumar** (88% match) - Data Scientist\n\n" \
                      "Would you like detailed screening reports for any of these?"
        
        elif any(word in prompt_lower for word in ['schedule', 'interview', 'meeting']):
            response = "📅 You have 8 interviews scheduled this week:\n\n" \
                      "**Today:**\n" \
                      "• Priya Sharma (Senior SWE) - 2:00 PM with Vikram Singh\n\n" \
                      "**Tomorrow:**\n" \
                      "• Rahul Kumar (Data Scientist) - 10:00 AM with Sneha Reddy\n\n" \
                      "I can auto-schedule the next batch if you approve the shortlist."
        
        elif any(word in prompt_lower for word in ['offer', 'drop', 'risk', 'engagement']):
            response = "⚠️ **High-Priority Alert:**\n\n" \
                      "**Ananya Iyer** (DevOps Engineer) shows 68% engagement score - HIGH risk of offer drop-off.\n\n" \
                      "**Indicators:**\n" \
                      "• Reduced response time\n" \
                      "• Competitor offer suspected\n" \
                      "• LinkedIn activity increased\n\n" \
                      "**Recommendations:**\n" \
                      "1. Expedite final decision (within 48 hours)\n" \
                      "2. Personalized outreach from hiring manager\n" \
                      "3. Consider salary adjustment (+8-12%)"
        
        elif any(word in prompt_lower for word in ['insight', 'analytics', 'trend', 'data']):
            response = "📈 **Key Hiring Insights:**\n\n" \
                      "**Performance Metrics:**\n" \
                      "• Average time-to-hire: 18 days (↓15% from last month)\n" \
                      "• Offer acceptance rate: 88%\n" \
                      "• Quality of hire score: 85%\n\n" \
                      "**Trends:**\n" \
                      "• Python skills are in highest demand (87 candidates)\n" \
                      "• 78% of candidates prefer hybrid roles\n" \
                      "• LinkedIn is the best source (45% conversion)\n\n" \
                      "**Predictions:**\n" \
                      "• Expected 18 offers this month\n" \
                      "• DevOps roles need salary adjustment"
        
        elif any(word in prompt_lower for word in ['skill', 'python', 'aws', 'react']):
            response = "🎯 **Top Skills Analysis:**\n\n" \
                      "**Most In-Demand:**\n" \
                      "1. Python - 87 candidates (65% match rate)\n" \
                      "2. AWS - 65 candidates (58% match rate)\n" \
                      "3. React - 52 candidates (71% match rate)\n" \
                      "4. Machine Learning - 48 candidates (55% match rate)\n\n" \
                      "**Skill Gaps:**\n" \
                      "• Kubernetes expertise is rare\n" \
                      "• Senior DevOps roles hardest to fill\n\n" \
                      "Would you like to see candidates with specific skills?"
        
        else:
            response = "I can help you with:\n\n" \
                      "• 🔍 **Resume screening** - Find top candidates by skills\n" \
                      "• 📅 **Interview scheduling** - Manage your calendar\n" \
                      "• 📊 **Candidate analytics** - Track pipeline health\n" \
                      "• ⚠️ **Risk prediction** - Identify offer drop-off risks\n" \
                      "• 📈 **Hiring insights** - Data-driven recommendations\n\n" \
                      "What would you like to explore?"
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>"
    "🧠 GCC Hiring Intelligence Platform | Built for GCC X Shift Hackathon 2025 | "
    "<strong>Multi-Tenant • AI-Powered • Scalable</strong>"
    "</div>",
    unsafe_allow_html=True
)
