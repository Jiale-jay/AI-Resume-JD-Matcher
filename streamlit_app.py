import os
import tempfile
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


# =========================
# LLM
# =========================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# =========================
# Helpers
# =========================
def load_pdf_text(file_path: str) -> str:
    text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def evaluate_match(resume: str, jd: str) -> str:
    prompt = f"""
You are an AI career assistant.

Compare the following resume and job description.

Resume:
{resume}

Job Description:
{jd}

Tasks:
1. Give a match score (0-100)
2. List key strengths
3. List missing skills
4. Give specific improvement suggestions

Use clear headings.
Be concise and structured.
"""
    response = llm.invoke(prompt)
    return response.content


def improve_resume(resume: str, jd: str) -> str:
    prompt = f"""
You are an expert in resume optimization.

Rewrite or improve the resume content so it aligns better with the job description.

Resume:
{resume}

Job Description:
{jd}

Requirements:
- Keep everything truthful
- Do not invent experience
- Focus on wording, positioning, and emphasis
- Make the result professional and concise
"""
    response = llm.invoke(prompt)
    return response.content


# =========================
# UI
# =========================
st.set_page_config(
    page_title="AI Resume-JD Matcher",
    page_icon="📄",
    layout="wide"
)

st.title("AI Resume-JD Matcher")
st.caption("Upload a resume PDF and compare it against a job description using LLMs.")

with st.sidebar:
    st.header("About")
    st.write(
        "This demo analyzes the match between a candidate resume and a job description, "
        "then suggests improvements using an LLM."
    )
    st.write("Built with Streamlit + OpenAI API.")

uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description", height=300, placeholder="Paste the job description here...")


col1, col2 = st.columns(2)

with col1:
    analyze_clicked = st.button("Analyze Match", use_container_width=True)

with col2:
    improve_clicked = st.button("Generate Resume Suggestions", use_container_width=True)


if uploaded_resume is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_resume.read())
        temp_pdf_path = tmp_file.name

    try:
        resume_text = load_pdf_text(temp_pdf_path)
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
else:
    resume_text = ""


if analyze_clicked:
    if not uploaded_resume:
        st.warning("Please upload a resume PDF.")
    elif not job_description.strip():
        st.warning("Please paste a job description.")
    elif not resume_text.strip():
        st.error("Could not extract text from the uploaded PDF.")
    else:
        with st.spinner("Analyzing resume-job match..."):
            try:
                result = evaluate_match(resume_text[:12000], job_description[:12000])
                st.subheader("Match Analysis")
                st.write(result)
            except Exception as e:
                st.error(f"Analysis failed: {e}")


if improve_clicked:
    if not uploaded_resume:
        st.warning("Please upload a resume PDF.")
    elif not job_description.strip():
        st.warning("Please paste a job description.")
    elif not resume_text.strip():
        st.error("Could not extract text from the uploaded PDF.")
    else:
        with st.spinner("Generating resume suggestions..."):
            try:
                improved = improve_resume(resume_text[:12000], job_description[:12000])
                st.subheader("Resume Improvement Suggestions")
                st.write(improved)
            except Exception as e:
                st.error(f"Resume improvement failed: {e}")


with st.expander("Preview Extracted Resume Text"):
    if resume_text:
        st.text_area("Extracted Text", resume_text[:5000], height=250)
    else:
        st.info("Upload a resume PDF to preview extracted text.")


def chat_with_resume_context(resume: str, jd: str, question: str) -> str:
    prompt = f"""
You are an AI career assistant.

You have access to:
1. The candidate's resume
2. The target job description

Resume:
{resume}

Job Description:
{jd}

User Question:
{question}

Answer clearly and practically.
Keep the answer concise but useful.
If relevant, explain how the candidate can improve alignment with the job.
"""
    response = llm.invoke(prompt)
    return response.content

st.markdown("## 💬 Resume Chat Assistant")

chat_question = st.text_input(
    "Ask a question about your resume and the job description",
    placeholder="e.g. What are my biggest gaps for this role?"
)

if st.button("Ask Assistant", use_container_width=True):
    if not uploaded_resume:
        st.warning("Please upload a resume PDF.")
    elif not job_description.strip():
        st.warning("Please paste a job description.")
    elif not resume_text.strip():
        st.error("Could not extract text from the uploaded PDF.")
    elif not chat_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = chat_with_resume_context(
                    resume_text[:12000],
                    job_description[:12000],
                    chat_question
                )
                st.session_state.chat_history.append(
                    {"question": chat_question, "answer": answer}
                )
            except Exception as e:
                st.error(f"Chat failed: {e}")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if st.session_state.chat_history:
    st.markdown("### Chat History")
    for item in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {item['question']}")
        st.markdown(f"**Assistant:** {item['answer']}")
        st.markdown("---")