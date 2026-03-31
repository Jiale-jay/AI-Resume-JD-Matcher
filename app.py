import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType

from jd_agent import evaluate_match, improve_resume


load_dotenv()

DATA_DIR = "data"


def load_all_pdfs(data_dir: str):
    pdf_dir = Path(data_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {data_dir}")

    all_docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = pdf_file.name
        all_docs.extend(docs)

    return all_docs


def build_vectorstore(data_dir: str):
    documents = load_all_pdfs(data_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


vectorstore = build_vectorstore(DATA_DIR)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


@tool
def retrieve_relevant_context(question: str) -> str:
    """
    Retrieve relevant information from indexed PDF documents.
    Use this tool when the user asks about document content.
    """
    docs = retriever.invoke(question)
    if not docs:
        return "No relevant content found."

    results = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown_file")
        page = doc.metadata.get("page", "unknown_page")
        text = doc.page_content[:1000]
        results.append(f"[Source {i}] File: {source}, Page: {page}\n{text}")

    return "\n\n".join(results)


@tool
def summarize_question_intent(question: str) -> str:
    """
    Summarize the user's question into a short intent description.
    Useful before answering complex or multi-part questions.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"Summarize the user's intent in one sentence:\n\n{question}"
    response = llm.invoke(prompt)
    return response.content


from langchain_community.document_loaders import PyPDFLoader


def load_pdf_text(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])


def run_jd_agent():
    resume_pdf = Path("data/resume.pdf")
    jd_path = Path("data/jd.txt")

    if not resume_pdf.exists():
        print("Error: data/resume.pdf not found.")
        return

    if not jd_path.exists():
        print("Error: data/jd.txt not found.")
        return

    #  PDF
    resume = load_pdf_text(str(resume_pdf))

    # JD 
    with open(jd_path, "r", encoding="utf-8") as f:
        jd = f.read()

    print("\n--- MATCH ANALYSIS ---\n")
    result = evaluate_match(resume, jd)
    print(result)

    print("\n--- IMPROVED RESUME ---\n")
    improved = improve_resume(resume, jd)
    print(improved)


def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    tools = [
        retrieve_relevant_context,
        summarize_question_intent,
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    chat_history = []

    print("Multi-PDF Agent + RAG system is ready.")
    print("Commands:")
    print("  analyze  -> run resume vs JD matching")
    print("  exit     -> quit")
    print("Otherwise, ask questions about your PDF documents.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        if user_input.lower() == "analyze":
            run_jd_agent()
            continue

        try:
            history_text = ""
            if chat_history:
                history_text = "\n".join(
                    [f"User: {q}\nAssistant: {a}" for q, a in chat_history[-3:]]
                )

            full_input = f"""
You are answering questions based on document content.

Recent conversation:
{history_text}

Current user question:
{user_input}
""".strip()

            response = agent.run(full_input)
            print(f"\nAgent: {response}\n")

            chat_history.append((user_input, response))

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()