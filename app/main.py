from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.ingestion import extract_text, chunk_text
from app.vectorstore import add_chunks
from app.rag import generate_answer

app = FastAPI(title="DocuHelp")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    source: str
    messages: list[Message]
    provider: str = "groq"


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_bytes = await file.read()
    try:
        text = extract_text(file.filename, file_bytes)
        chunks = chunk_text(text)
        count = add_chunks(doc_id=file.filename, chunks=chunks)
        return {"filename": file.filename, "chunks_stored": count}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/query")
def query(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    return generate_answer(messages, source=req.source, provider=req.provider)


@app.get("/")
def home():
    return FileResponse("static/index.html")