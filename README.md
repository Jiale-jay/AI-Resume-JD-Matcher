
#  AI Resume-JD Matcher

An end-to-end AI application that analyzes the match between a candidate's resume and a job description using Large Language Models (LLMs), and provides actionable improvement suggestions.

---

##  Features

*  Upload resume in PDF format
*  Input job description
*  Match score evaluation (0–100)
*  Strengths analysis
*  Skill gap identification
*  Resume improvement suggestions
*  LLM-powered reasoning
*  Interactive UI built with Streamlit

---

##  Tech Stack

* **LLM**: OpenAI (GPT-4o-mini)
* **Framework**: LangChain
* **Frontend**: Streamlit
* **PDF Parsing**: pdfplumber
* **Language**: Python

---

##  System Architecture

```
User Input
   │
   ├── Resume (PDF)
   ├── Job Description (Text)
   │
   ▼
PDF Parsing (pdfplumber)
   ▼
Text Processing
   ▼
LLM (GPT-4o-mini)
   │
   ├── Match Score
   ├── Strengths
   ├── Missing Skills
   └── Suggestions
   ▼
Streamlit UI Visualization
```

---

##  Installation

```bash
git clone https://github.com/yourusername/ai-resume-matcher.git
cd ai-resume-matcher

pip install -r requirements.txt
```

---

##  Environment Setup

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## ▶ Run the App

```bash
streamlit run streamlit_app.py
```

---

##  How to Use

1. Upload your **resume (PDF)**
2. Paste the **job description**
3. Click:

   * `Analyze Match` → get score + analysis
   * `Generate Resume Suggestions` → get optimized content

---

##  Example Output

* Match Score: **78/100**
* Strengths:

  * Strong background in computer vision
  * Experience with model deployment
* Missing Skills:

  * LLM / Agent systems
* Suggestions:

  * Add an LLM-based project
  * Improve wording for impact

---

##  Key Highlights

* Designed structured LLM prompts for consistent output parsing
* Built an end-to-end AI workflow (not just a model demo)
* Integrated PDF parsing + LLM reasoning + UI visualization
* Focused on **real-world business use case (recruitment / consulting)**

---

##  Future Improvements

* Upload JD as PDF
* Multi-job comparison
* Export optimized resume
* History tracking
* Deployment (Streamlit Cloud / Docker)

---

##  Author

**Jiale Guo (Jay)**
MSc Electronic & Computer Engineering @ DCU

* Computer Vision
* AI Systems
* Neural Rendering

---

##  Notes

This project is designed as a **practical AI application** demonstrating how LLMs can be integrated into real-world workflows beyond simple chat interfaces.
