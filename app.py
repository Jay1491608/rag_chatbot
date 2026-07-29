from fastapi import FastAPI
from rag_full import ask_rag as ask_rag_fn
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI(
    title="RAG API",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str
    
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/ask")
def ask_question(data: QuestionRequest):
    try:
        answer = ask_rag_fn(data.question)
        return {
            "success": True,
            "question": data.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
