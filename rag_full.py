from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS #Facebook AI Similarity Search
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
# import sqlite3
from pathlib import Path
from config import API_KEY, BASE_URL, MODEL_NAME


BASE_DIR =  Path(__file__).parent
file = BASE_DIR / "system_design.txt"


# ---------------------------
# Read TXT File
# ---------------------------
with open(file, "r", encoding="utf-8") as f:
    text = f.read()

# conn = sqlite3.connect('company.db')
# cursor = conn.cursor()

# # Read data from the table
# cursor.execute("SELECT * FROM employees")
# rows = cursor.fetchall()

# ---------------------------
# Split Document
# ---------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)
docs = splitter.create_documents([text]) 

# ---------------------------
# Embedding Model
# ---------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------
# Create Vector DB
# ---------------------------
vector_db = FAISS.from_documents(
    docs,
    embedding_model
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)

# --------------------------
# LLM
# ---------------------------
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key= API_KEY,
    base_url=BASE_URL
)

# ---------------------------
# RAG Chain
# ---------------------------
system_prompt = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
# prompt = PromptTemplate(
# input_variables=["context", "question"],
# template= system_prompt)

question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
    
)
qa_chain = create_retrieval_chain(retriever, question_answer_chain )


def ask_rag(question: str):
    
    result = qa_chain.invoke(
        {
            "input":question
        }
    )
 
    return result["answer"]
    
    