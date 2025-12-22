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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

def get_chatbot_response(user_question):
    question_lower = user_question.lower()
    
    rule_based_response = None
    
    if "status" in question_lower or "where is my resume" in question_lower:
        if st.session_state.candidates_data:
            rule_based_response = "Your resume has been screened. Check the screening results section to see your score and status."
        else:
            rule_based_response = "No resumes have been uploaded yet. Please wait for the screening process to begin."
    
    elif "interview" in question_lower and "status" in question_lower:
        if st.session_state.interview_scores:
            rule_based_response = "Interview scores have been recorded. Check the final rankings section for your evaluation."
        else:
            rule_based_response = "No interview evaluations have been completed yet."
    
    elif "review" in question_lower and "mean" in question_lower:
        rule_based_response = "REVIEW means your application shows potential but needs further evaluation through an interview before a final decision is made."
    
    elif "improve" in question_lower or "future" in question_lower:
        rule_based_response = "Focus on building relevant technical skills, gaining more experience, and tailoring your resume to match job requirements. Check the feedback section for specific comments."
    
    elif "hire" in question_lower and "mean" in question_lower:
        rule_based_response = "HIRE means your application has been approved and you are being recommended for the position."
    
    elif "reject" in question_lower and "mean" in question_lower:
        rule_based_response = "REJECT means your application does not meet the current requirements. You can apply again in the future with updated skills."
    
    elif "process" in question_lower or "how does" in question_lower:
        rule_based_response = "The hiring process involves: 1) Resume screening, 2) Interview evaluation, 3) Final decision by the hiring team."
    
    elif "feedback" in question_lower or "comment" in question_lower:
        rule_based_response = "You can view and leave feedback in the Comments section at the bottom of the page."
    
    if rule_based_response:
        if st.session_state.groq_client:
            try:
                prompt = f"""Rephrase this message in a friendly, professional tone:
{rule_based_response}

Keep it concise and helpful."""
                response = st.session_state.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=100
                )
                return response.choices[0].message.content.strip()
            except:
                return rule_based_response
        else:
            return rule_based_response
    else:
        if st.session_state.groq_client:
            try:
                context = f"""You are a helpful hiring assistant. Answer questions about the hiring process.
Current system state:
- Candidates screened: {len(st.session_state.candidates_data)}
- Interviews completed: {len(st.session_state.interview_scores)}
- Final decisions made: {len(st.session_state.final_decisions)}

User question: {user_question}

Provide a helpful, brief response. Do not make up specific candidate information."""
                
                response = st.session_state.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": context}],
                    temperature=0.7,
                    max_tokens=150
                )
                return response.choices[0].message.content.strip()
            except:
                return "I'm here to help with questions about the hiring process. You can ask about your status, what different decisions mean, or how to improve."
        else:
            return "I'm here to help with questions about the hiring process. You can ask about your status, what different decisions mean, or how to improve."

st.markdown('<p class="main-header">🎯 GCC AI Hiring Platform</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📋 Hiring Process", "💬 Chatbot", "💭 Feedback"])

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
    st.markdown('<p class="section-header">💬 Hiring Assistant Chatbot</p>', unsafe_allow_html=True)
    st.write("Ask questions about the hiring process, your status, or how to improve")
    
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")
    
    user_input = st.text_input("Ask a question", key="chat_input", placeholder="e.g., Where is my resume now?")
    
    if st.button("Send", key="send_chat"):
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            bot_response = get_chatbot_response(user_input)
            
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            st.rerun()

with tab3:
    st.markdown('<p class="section-header">💭 Feedback & Comments</p>', unsafe_allow_html=True)
    st.write("Share feedback or comments about the hiring process")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        commenter_name = st.text_input("Your Name", placeholder="Enter your name")
    
    with col2:
        commenter_role = st.selectbox("Role", ["Candidate", "Interviewer", "Recruiter"])
    
    comment_text = st.text_area("Comment", placeholder="Share your feedback here...", height=100)
    
    if st.button("Post Comment"):
        if commenter_name and comment_text:
            st.session_state.feedback_comments.append({
                "name": commenter_name,
                "role": commenter_role,
                "comment": comment_text,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("Comment posted!")
            st.rerun()
        else:
            st.error("Please enter your name and comment")
    
    st.markdown("---")
    st.markdown("### All Comments")
    
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
        st.info("No comments yet. Be the first to share feedback!")
