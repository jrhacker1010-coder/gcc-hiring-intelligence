# 🎯 GCC AI Hiring Platform  
**GCC X-Shift Hackathon 2025**  
**Problem Option: Hiring in Global Capability Centers (GCCs)**

---

## 📖 Introduction

Global Capability Centers (GCCs) play a key role in large-scale hiring across technology, operations, and support roles.  
However, many GCC hiring workflows still depend on manual resume screening, Excel-based tracking, and unstructured interview feedback.

This project is a **simple AI-assisted hiring platform** that helps recruiters screen candidates faster while maintaining **human control, transparency, and fairness**.

The solution is designed as a **hackathon prototype**, focusing on clarity and practicality rather than complex or black-box AI.

---

## ❗ Problem Statement

Hiring teams in GCCs commonly face the following challenges:

- Manual resume shortlisting takes significant time
- No consistent method to compare resumes with job descriptions
- Interview feedback is often scattered and subjective
- Candidates do not understand their application status
- Limited transparency in hiring decisions

These issues slow down the hiring process and negatively affect both recruiters and candidates.

---

## 💡 Proposed Solution

This project provides a **single platform** that:

- Automates resume screening using explainable logic
- Ranks candidates based on job relevance
- Supports structured interview evaluation
- Uses AI only as a **decision-support tool**
- Improves candidate experience through feedback and chatbot support

The system **assists humans**, not replaces them.

---

## 🧩 Functional Modules

### 🔹 Module 1: Resume Screening

- Recruiter enters a Job Description
- Uploads multiple candidate resumes (PDF)
- System:
  - Extracts resume text
  - Identifies relevant skills
  - Matches resume content with JD
  - Calculates a resume score
  - Ranks candidates automatically

---

### 🔹 Module 2: Interview Evaluation

- Interviewers rate candidates on:
  - Technical skills
  - Communication
  - Problem solving
  - Cultural fit
- Interview score is calculated
- Resume score and interview score are combined
- Final ranking is generated

---

### 🔹 Module 3: AI-Assisted Recommendation

- System suggests:
  - **HIRE**
  - **REVIEW**
  - **REJECT**
- Each recommendation includes a short explanation
- Final decision is always taken by the recruiter

---

### 🔹 Module 4: Candidate Experience & Transparency

- Candidates can:
  - Check application status
  - View feedback
  - Practice interview questions
  - Ask hiring-related questions via chatbot
- Improves trust and communication

---

### 🔹 Module 5: Role-Based Access

- **Admin Mode**
  - Resume screening
  - Candidate ranking
  - Interview evaluation
  - Final hiring decisions

- **Candidate Mode**
  - Interview preparation
  - Status tracking
  - Feedback viewing
  - Chatbot support

---

## 🧠 AI & Logic Used

The project intentionally uses **simple and explainable methods**:

- **TF-IDF + Cosine Similarity** for JD–resume matching
- **Rule-based scoring** for skills and experience
- **Large Language Model (Groq API)** used only for:
  - Explanations
  - Chatbot responses
  - Interview assistance

No automatic or hidden decisions are made by AI.

---

## 🏗️ System Architecture (High Level)
Admin / Candidate
↓
Streamlit Interface
↓
Resume & Interview Processing
↓
AI Assistance (Optional)
↓
Session-Based Data Storage


This lightweight architecture is suitable for a hackathon prototype.

---

## 🛠️ Technology Stack

| Layer | Technology |
|------|-----------|
| Language | Python |
| UI | Streamlit |
| Data Handling | pandas |
| PDF Parsing | pypdf |
| Text Matching | scikit-learn |
| AI Assistance | Groq API (LLM) |

---

## ▶️ How to Run the Application

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt

Step 2: Run the App
streamlit run app.py

Step 3: AI Configuration (Optional)

Add GROQ_API_KEY in Streamlit or Replit secrets.

The application works even without the API key using fallback logic.

🎯 Hackathon Alignment

This solution directly aligns with Option 1: Hiring in GCCs by:

Reducing manual resume screening

Supporting structured hiring decisions

Improving recruiter efficiency

Enhancing candidate transparency

Using AI responsibly with human oversight

⚠️ Limitations

Uses session-based storage (no database)

Designed as a prototype, not production-ready

Skill extraction is keyword-based

🚀 Future Enhancements

Database integration

Advanced skill extraction

Historical hiring insights

Multi-round interview workflows

Analytics dashboard for recruiters

👨‍💻 Author

Harsh
Participant – GCC X-Shift Hackathon 2025

🙌 Final Note

This project demonstrates how simple, explainable AI can meaningfully improve GCC hiring processes while keeping humans fully in control of decisions.


---

## 🏁 You Are 100% READY

✔ Complete  
✔ Professional  
✔ Human  
✔ Judge-friendly  
✔ Submission-safe  

If you want next:
- 🎥 Demo walkthrough script  
- 📊 PPT content from README  
- 🔥 One-line project tagline  

Just say the word 💪

