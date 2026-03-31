from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv() 

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


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

Be concise and structured.
"""
    response = llm.invoke(prompt)
    return response.content


def improve_resume(resume: str, jd: str) -> str:
    prompt = f"""
You are an expert in resume optimization.

Rewrite the resume to better match the job description.

Resume:
{resume}

Job Description:
{jd}

Make it more aligned with the role while keeping it truthful.
"""
    response = llm.invoke(prompt)
    return response.content