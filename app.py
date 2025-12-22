import streamlit as st
import pandas as pd
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from datetime import datetime

st.set_page_config(page_title="GCC AI Hiring Platform", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .candidate-card {
        background: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    .score-high { background: #dcfce7; color: #166534; }
    .score-medium { background: #fef3c7; color: #92400e; }
    .score-low { background: #fee2e2; color: #991b1b; }
    .comment-box {
        background: #f3f4f6;
        padding: 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #9ca3af;
    }
    .comment-author {
        font-weight: 600;
        color: #374151;
        font-size: 0.875rem;
    }
    .comment-time {
        color: #6b7280;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

if "groq_client" not in st.session_state:
    try:
        st.session_state.groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    except:
        st.session_state.groq_client = None

if "candidates_data" not in st.session_state:
    st.session_state.candidates_data = []

if "interview_scores" not in st.session_state:
    st.session_state.interview_scores = {}

if "final_decisions" not in st.session_state:
    st.session_state.final_decisions = {}

if "feedback_comments" not in st.session_state:
    st.session_state.feedback_comments = []

if "admin_chat_history" not in st.session_state:
    st.session_state.admin_chat_history = []

if "candidate_chat_history" not in st.session_state:
    st.session_state.candidate_chat_history = []

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_role" not in st.session_state:
    st.session_state.user_role = None

if "candidate_name" not in st.session_state:
    st.session_state.candidate_name = None

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "
        return text.strip()
    except:
        return ""

def extract_skills(text):
    common_skills = [
        "python", "java", "javascript", "sql", "react", "node", "angular",
        "machine learning", "data science", "aws", "azure", "docker", "kubernetes",
        "git", "agile", "scrum", "html", "css", "mongodb", "postgresql",
        "excel", "powerbi", "tableau", "r", "c++", "django", "flask",
        "tensorflow", "pytorch", "nlp", "computer vision", "rest api"
    ]
    text_lower = text.lower()
    found_skills = []
    for skill in common_skills:
        if skill in text_lower:
            found_skills.append(skill)
    return found_skills

def extract_experience_years(text):
    patterns = [
        r'(\d+)\+?\s*years?',
        r'(\d+)\+?\s*yrs?',
        r'experience:\s*(\d+)',
    ]
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        years.extend([int(m) for m in matches])
    return max(years) if years else 0

def calculate_text_similarity(jd_text, resume_text):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([jd_text, resume_text])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return round(similarity * 100, 2)
    except:
        return 0

def calculate_resume_score(jd_text, resume_text, jd_skills, resume_skills, experience):
    text_match = calculate_text_similarity(jd_text, resume_text)
    
    if len(jd_skills) > 0:
        skill_match = (len(set(resume_skills) & set(jd_skills)) / len(jd_skills)) * 100
    else:
        skill_match = 0
    
    if experience >= 5:
        exp_score = 100
    elif experience >= 3:
        exp_score = 75
    elif experience >= 1:
        exp_score = 50
    else:
        exp_score = 25
    
    final_score = (text_match * 0.4) + (skill_match * 0.4) + (exp_score * 0.2)
    return round(final_score, 2)

def get_ai_recommendation(jd_text, resume_text, score, matched_skills, missing_skills, experience):
    if st.session_state.groq_client is None:
        if score >= 70:
            return "HIRE", "Strong match based on resume score"
        elif score >= 50:
            return "REVIEW", "Moderate match, needs interview"
        else:
            return "REJECT", "Low match with job requirements"
    
    try:
        prompt = f"""You are a hiring assistant. Based on the following information, provide a hiring recommendation.

Job Description: {jd_text[:500]}

Resume Summary: {resume_text[:500]}

Match Score: {score}/100
Experience: {experience} years
Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}
Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}

Provide your recommendation in exactly this format:
Decision: [HIRE or REVIEW or REJECT]
Reason: [One sentence explanation]"""

        response = st.session_state.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        
        result = response.choices[0].message.content.strip()
        
        if "HIRE" in result.upper():
            decision = "HIRE"
        elif "REVIEW" in result.upper():
            decision = "REVIEW"
        else:
            decision = "REJECT"
        
        reason_match = re.search(r'Reason:\s*(.+)', result, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else "No specific reason provided"
        
        return decision, reason
    except:
        if score >= 70:
            return "HIRE", "Strong match based on resume score"
        elif score >= 50:
            return "REVIEW", "Moderate match, needs interview"
        else:
            return "REJECT", "Low match with job requirements"

def get_admin_chatbot_response(user_question):
    question_lower = user_question.lower()
    
    if "top" in question_lower and "candidate" in question_lower:
        if st.session_state.candidates_data:
            top_candidates = sorted(st.session_state.candidates_data, key=lambda x: x["score"], reverse=True)[:5]
            response = "Top 5 candidates:\n"
            for idx, c in enumerate(top_candidates):
                response += f"{idx+1}. {c['name']} - Score: {c['score']}%\n"
            return response
        else:
            return "No candidates have been screened yet."
    
    elif "why" in question_lower and "reject" in question_lower:
        for candidate in st.session_state.candidates_data:
            if candidate['name'].lower() in question_lower:
                return f"{candidate['name']} - Decision: {candidate['ai_decision']}, Reason: {candidate['ai_reason']}"
        return "Please specify the candidate name in your question."
    
    elif "explain" in question_lower or "why" in question_lower:
        for candidate in st.session_state.candidates_data:
            if candidate['name'].lower() in question_lower:
                return f"{candidate['name']} - AI Decision: {candidate['ai_decision']}, Reason: {candidate['ai_reason']}, Score: {candidate['score']}%"
        return "Please specify which candidate you want to know about."
    
    if st.session_state.groq_client:
        try:
            context = f"""You are an admin hiring assistant. Answer questions about the hiring process and candidates.
Current system state:
- Candidates screened: {len(st.session_state.candidates_data)}
- Interviews completed: {len(st.session_state.interview_scores)}
- Final decisions made: {len(st.session_state.final_decisions)}

User question: {user_question}

Provide a helpful, brief response for the admin."""
            
            response = st.session_state.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": context}],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except:
            return "I'm here to help with admin questions about candidates and hiring decisions."
    else:
        return "I'm here to help with admin questions about candidates and hiring decisions."

def get_candidate_chatbot_response(user_question, candidate_name):
    question_lower = user_question.lower()
    
    candidate_data = None
    for c in st.session_state.candidates_data:
        if c['name'] == candidate_name:
            candidate_data = c
            break
    
    if "status" in question_lower or "application" in question_lower:
        if candidate_data:
            decision = st.session_state.final_decisions.get(candidate_name, "PENDING")
            return f"Your application status is: {decision}. Your resume score is {candidate_data['score']}%."
        else:
            return "Your application is being reviewed. Please check back later."
    
    elif "improve" in question_lower:
        if candidate_data:
            if candidate_data['missing_skills']:
                return f"To improve, consider developing these skills: {', '.join(candidate_data['missing_skills'][:5])}."
            else:
                return "Your skills match well! Focus on interview preparation and communication skills."
        else:
            return "Focus on building relevant technical skills and gaining experience in your field."
    
    elif "next" in question_lower or "process" in question_lower:
        return "The hiring process includes: 1) Resume screening, 2) Interview evaluation, 3) Final decision. You'll be notified at each stage."
    
    if st.session_state.groq_client:
        try:
            context = f"""You are a helpful assistant for job candidates. Answer questions about the hiring process from a candidate perspective.

User question: {user_question}

Provide a helpful, brief response. Do not answer admin or recruiter questions."""
            
            response = st.session_state.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": context}],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except:
            return "I'm here to help with questions about your application and the hiring process."
    else:
        return "I'm here to help with questions about your application and the hiring process."

def generate_interview_questions(candidate_name):
    candidate_data = None
    for c in st.session_state.candidates_data:
        if c['name'] == candidate_name:
            candidate_data = c
            break
    
    if not candidate_data:
        return ["Tell me about yourself.", "What are your strengths?", "Why do you want this position?"]
    
    if st.session_state.groq_client:
        try:
            prompt = f"""Generate 4 interview questions for a candidate with these skills: {', '.join(candidate_data['matched_skills'][:5])} and {candidate_data['experience']} years of experience.

Provide only the questions, numbered 1-4."""
            
            response = st.session_state.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=200
            )
            
            questions_text = response.choices[0].message.content.strip()
            questions = [q.strip() for q in questions_text.split('\n') if q.strip() and any(char.isdigit() for char in q[:3])]
            return questions[:4] if questions else ["Tell me about yourself.", "What are your strengths?", "Why do you want this position?"]
        except:
            return ["Tell me about yourself.", "What are your strengths?", "Why do you want this position?", "Describe a challenging project you worked on."]
    else:
        return ["Tell me about yourself.", "What are your strengths?", "Why do you want this position?", "Describe a challenging project you worked on."]

def evaluate_practice_answer(question, answer):
    if st.session_state.groq_client:
        try:
            prompt = f"""Provide brief feedback on this interview answer (2-3 sentences):

Question: {question}
Answer: {answer}

Give constructive feedback on clarity, relevance, and completeness."""
            
            response = st.session_state.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except:
            return "Good effort! Focus on being specific and providing concrete examples."
    else:
        return "Good effort! Focus on being specific and providing concrete examples."

if not st.session_state.logged_in:
    st.markdown('<p class="main-header">🎯 GCC AI Hiring Platform</p>', unsafe_allow_html=True)
    st.markdown("### Select Your Role")
    
    role_choice = st.radio("I am a:", ["Admin", "Candidate"])
    
    if role_choice == "Admin":
        st.markdown("#### Admin Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username == "admin" and password == "123":
                st.session_state.logged_in = True
                st.session_state.user_role = "admin"
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    else:
        st.markdown("#### Candidate Login")
        candidate_name = st.text_input("Enter your name (as on resume)")
        
        if st.button("Continue"):
            if candidate_name:
                st.session_state.logged_in = True
                st.session_state.user_role = "candidate"
                st.session_state.candidate_name = candidate_name
                st.rerun()
            else:
                st.error("Please enter your name")

else:
    st.markdown('<p class="main-header">🎯 GCC AI Hiring Platform</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([6, 1])
    with col1:
        st.write(f"**Logged in as:** {st.session_state.user_role.upper()}")
        if st.session_state.user_role == "candidate":
            st.write(f"**Name:** {st.session_state.candidate_name}")
    with col2:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_role = None
            st.session_state.candidate_name = None
            st.rerun()
    
    if st.session_state.user_role == "admin":
        tab1, tab2, tab3 = st.tabs(["📋 Hiring Process", "💬 Admin Chatbot", "💭 Feedback"])
        
        with tab1:
            st.markdown("### 📋 Step 1: Job Description")
            job_description = st.text_area(
                "Paste the job description here",
                height=200,
                placeholder="Enter the full job description including required skills, experience, and responsibilities..."
            )

            st.markdown("### 📄 Step 2: Upload Resumes")
            uploaded_files = st.file_uploader(
                "Upload candidate resumes (PDF format)",
                type=["pdf"],
                accept_multiple_files=True
            )

            if st.button("🔍 Screen Candidates", type="primary"):
                if not job_description:
                    st.error("Please enter a job description")
                elif not uploaded_files:
                    st.error("Please upload at least one resume")
                else:
                    with st.spinner("Analyzing resumes..."):
                        jd_skills = extract_skills(job_description)
                        candidates = []
                        
                        for uploaded_file in uploaded_files:
                            resume_text = extract_text_from_pdf(uploaded_file)
                            
                            if not resume_text:
                                continue
                            
                            resume_skills = extract_skills(resume_text)
                            experience = extract_experience_years(resume_text)
                            missing_skills = list(set(jd_skills) - set(resume_skills))
                            
                            score = calculate_resume_score(
                                job_description,
                                resume_text,
                                jd_skills,
                                resume_skills,
                                experience
                            )
                            
                            ai_decision, ai_reason = get_ai_recommendation(
                                job_description,
                                resume_text,
                                score,
                                resume_skills,
                                missing_skills,
                                experience
                            )
                            
                            candidates.append({
                                "name": uploaded_file.name.replace(".pdf", ""),
                                "score": score,
                                "experience": experience,
                                "matched_skills": resume_skills,
                                "missing_skills": missing_skills,
                                "ai_decision": ai_decision,
                                "ai_reason": ai_reason,
                                "resume_text": resume_text
                            })
                        
                        candidates.sort(key=lambda x: x["score"], reverse=True)
                        st.session_state.candidates_data = candidates
                        
                        st.success(f"✅ Screened {len(candidates)} candidates successfully!")

            if st.session_state.candidates_data:
                st.markdown('<p class="section-header">📊 Screening Results</p>', unsafe_allow_html=True)
                
                results_df = pd.DataFrame([
                    {
                        "Rank": idx + 1,
                        "Candidate": c["name"],
                        "Resume Score": f"{c['score']}%",
                        "Experience": f"{c['experience']} yrs",
                        "AI Decision": c["ai_decision"],
                        "AI Reason": c["ai_reason"],
                        "Matched Skills": len(c["matched_skills"]),
                        "Missing Skills": len(c["missing_skills"])
                    }
                    for idx, c in enumerate(st.session_state.candidates_data)
                ])
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                st.markdown('<p class="section-header">🎤 Interview Evaluation</p>', unsafe_allow_html=True)
                
                candidate_names = [c["name"] for c in st.session_state.candidates_data]
                selected_candidate = st.selectbox("Select candidate to evaluate", candidate_names)
                
                if selected_candidate:
                    candidate_data = next(c for c in st.session_state.candidates_data if c["name"] == selected_candidate)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Resume Score:** {candidate_data['score']}%")
                        st.write(f"**Experience:** {candidate_data['experience']} years")
                        st.write(f"**AI Decision:** {candidate_data['ai_decision']}")
                    
                    with col2:
                        st.write(f"**Matched Skills:** {', '.join(candidate_data['matched_skills'][:5]) if candidate_data['matched_skills'] else 'None'}")
                        st.write(f"**AI Reason:** {candidate_data['ai_reason']}")
                    
                    st.markdown("#### Rate the candidate on interview performance")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        technical = st.slider("Technical Skills", 1, 10, 5, key=f"tech_{selected_candidate}")
                        communication = st.slider("Communication", 1, 10, 5, key=f"comm_{selected_candidate}")
                    
                    with col2:
                        problem_solving = st.slider("Problem Solving", 1, 10, 5, key=f"prob_{selected_candidate}")
                        cultural_fit = st.slider("Cultural Fit", 1, 10, 5, key=f"cult_{selected_candidate}")
                    
                    interview_score = round((technical * 0.4 + communication * 0.25 + problem_solving * 0.25 + cultural_fit * 0.1) * 10, 2)
                    
                    st.metric("Interview Score", f"{interview_score}%")
                    
                    if st.button("💾 Save Interview Score"):
                        st.session_state.interview_scores[selected_candidate] = interview_score
                        st.success(f"Interview score saved for {selected_candidate}")
                
                st.markdown('<p class="section-header">🏆 Final Rankings</p>', unsafe_allow_html=True)
                
                final_rankings = []
                for candidate in st.session_state.candidates_data:
                    name = candidate["name"]
                    resume_score = candidate["score"]
                    interview_score = st.session_state.interview_scores.get(name, 0)
                    
                    if interview_score > 0:
                        final_score = round((resume_score * 0.5) + (interview_score * 0.5), 2)
                    else:
                        final_score = resume_score
                    
                    final_rankings.append({
                        "Candidate": name,
                        "Resume Score": resume_score,
                        "Interview Score": interview_score,
                        "Final Score": final_score,
                        "Decision": st.session_state.final_decisions.get(name, "PENDING")
                    })
                
                final_rankings.sort(key=lambda x: x["Final Score"], reverse=True)
                
                for idx, candidate in enumerate(final_rankings):
                    candidate["Final Rank"] = idx + 1
                
                final_df = pd.DataFrame(final_rankings)
                st.dataframe(final_df, use_container_width=True, hide_index=True)
                
                st.markdown('<p class="section-header">✅ Human Decision</p>', unsafe_allow_html=True)
                st.write("Review each candidate and make the final hiring decision")
                
                for candidate in final_rankings:
                    with st.expander(f"Rank #{candidate['Final Rank']}: {candidate['Candidate']} (Final Score: {candidate['Final Score']}%)"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Resume", f"{candidate['Resume Score']}%")
                        with col2:
                            st.metric("Interview", f"{candidate['Interview Score']}%")
                        with col3:
                            st.metric("Final", f"{candidate['Final Score']}%")
                        
                        st.write(f"**Current Decision:** {candidate['Decision']}")
                        
                        decision_col1, decision_col2, decision_col3 = st.columns(3)
                        
                        with decision_col1:
                            if st.button("✅ HIRE", key=f"hire_{candidate['Candidate']}"):
                                st.session_state.final_decisions[candidate['Candidate']] = "HIRE"
                                st.rerun()
                        
                        with decision_col2:
                            if st.button("🔄 REVIEW", key=f"review_{candidate['Candidate']}"):
                                st.session_state.final_decisions[candidate['Candidate']] = "REVIEW"
                                st.rerun()
                        
                        with decision_col3:
                            if st.button("❌ REJECT", key=f"reject_{candidate['Candidate']}"):
                                st.session_state.final_decisions[candidate['Candidate']] = "REJECT"
                                st.rerun()
        
        with tab2:
            st.markdown('<p class="section-header">💬 Admin Chatbot</p>', unsafe_allow_html=True)
            st.write("Ask questions about candidates, decisions, and hiring process")
            
            for msg in st.session_state.admin_chat_history:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**Assistant:** {msg['content']}")
            
            user_input = st.text_input("Ask a question", key="chat_input", placeholder="e.g., Top 5 candidates?")
            
            if st.button("Send", key="send_chat"):
                if user_input:
                    st.session_state.admin_chat_history.append({"role": "user", "content": user_input})
                    
                    bot_response = get_admin_chatbot_response(user_input)
                    
                    st.session_state.admin_chat_history.append({"role": "assistant", "content": bot_response})
                    st.rerun()
        
        with tab3:
            st.markdown('<p class="section-header">💭 All Feedback</p>', unsafe_allow_html=True)
            
            if st.session_state.feedback_comments:
                for comment in reversed(st.session_state.feedback_comments):
                    st.markdown(f"""
                    <div class="comment-box">
                        <span class="comment-author">{comment['name']}</span> 
                        <span style="color: #6b7280;">({comment['role']})</span>
                        <span class="comment-time"> • {comment['timestamp']}</span>
                        <p style="margin-top: 0.5rem; margin-bottom: 0;">{comment['comment']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No feedback yet.")
    
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 My Status", "💬 Candidate Chatbot", "💭 My Feedback", "🎯 Interview Practice"])
        
        with tab1:
            st.markdown('<p class="section-header">📊 Application Status</p>', unsafe_allow_html=True)
            
            candidate_data = None
            for c in st.session_state.candidates_data:
                if c['name'] == st.session_state.candidate_name:
                    candidate_data = c
                    break
            
            if candidate_data:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Resume Score", f"{candidate_data['score']}%")
                with col2:
                    interview_score = st.session_state.interview_scores.get(st.session_state.candidate_name, 0)
                    st.metric("Interview Score", f"{interview_score}%")
                with col3:
                    decision = st.session_state.final_decisions.get(st.session_state.candidate_name, "PENDING")
                    st.metric("Status", decision)
                
                st.markdown("#### Your Details")
                st.write(f"**Experience:** {candidate_data['experience']} years")
                st.write(f"**Matched Skills:** {', '.join(candidate_data['matched_skills']) if candidate_data['matched_skills'] else 'None'}")
                st.write(f"**Skills to Develop:** {', '.join(candidate_data['missing_skills']) if candidate_data['missing_skills'] else 'None'}")
                st.write(f"**AI Recommendation:** {candidate_data['ai_decision']}")
            else:
                st.info("Your application is being reviewed. Please check back later.")
        
        with tab2:
            st.markdown('<p class="section-header">💬 Candidate Chatbot</p>', unsafe_allow_html=True)
            st.write("Ask questions about your application and the hiring process")
            
            for msg in st.session_state.candidate_chat_history:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**Assistant:** {msg['content']}")
            
            user_input = st.text_input("Ask a question", key="chat_input", placeholder="e.g., What is my status?")
            
            if st.button("Send", key="send_chat"):
                if user_input:
                    st.session_state.candidate_chat_history.append({"role": "user", "content": user_input})
                    
                    bot_response = get_candidate_chatbot_response(user_input, st.session_state.candidate_name)
                    
                    st.session_state.candidate_chat_history.append({"role": "assistant", "content": bot_response})
                    st.rerun()
        
        with tab3:
            st.markdown('<p class="section-header">💭 Share Your Feedback</p>', unsafe_allow_html=True)
            
            comment_text = st.text_area("Your feedback", placeholder="Share your thoughts about the hiring process...", height=100)
            
            if st.button("Post Feedback"):
                if comment_text:
                    st.session_state.feedback_comments.append({
                        "name": st.session_state.candidate_name,
                        "role": "Candidate",
                        "comment": comment_text,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.success("Feedback posted!")
                    st.rerun()
                else:
                    st.error("Please enter your feedback")
            
            st.markdown("---")
            st.markdown("### Your Previous Feedback")
            
            my_comments = [c for c in st.session_state.feedback_comments if c['name'] == st.session_state.candidate_name]
            
            if my_comments:
                for comment in reversed(my_comments):
                    st.markdown(f"""
                    <div class="comment-box">
                        <span class="comment-author">{comment['name']}</span>
                        <span class="comment-time"> • {comment['timestamp']}</span>
                        <p style="margin-top: 0.5rem; margin-bottom: 0;">{comment['comment']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("You haven't posted any feedback yet.")
        
        with tab4:
            st.markdown('<p class="section-header">🎯 Interview Practice</p>', unsafe_allow_html=True)
            st.write("Practice answering interview questions and get AI feedback")
            
            if st.button("Generate Practice Questions"):
                questions = generate_interview_questions(st.session_state.candidate_name)
                st.session_state.practice_questions = questions
                st.rerun()
            
            if "practice_questions" in st.session_state:
                st.markdown("#### Your Practice Questions")
                
                for idx, question in enumerate(st.session_state.practice_questions):
                    st.markdown(f"**{question}**")
                    answer = st.text_area(f"Your answer", key=f"answer_{idx}", height=100, placeholder="Type your answer here...")
                    
                    if st.button(f"Get Feedback", key=f"feedback_{idx}"):
                        if answer:
                            feedback = evaluate_practice_answer(question, answer)
                            st.info(feedback)
                        else:
                            st.warning("Please write an answer first")
                    
                    st.markdown("---")
