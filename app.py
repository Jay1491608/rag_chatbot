from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag_service import RAGService, get_rag_service


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.rag_service = get_rag_service()
    yield


app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)


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
def ask_question(data: QuestionRequest, request: Request):
    try:
        service: RAGService = request.app.state.rag_service
        answer = service.ask(data.question)
        return {
            "success": True,
            "question": data.question,
            "answer": answer,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
