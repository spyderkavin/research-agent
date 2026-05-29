from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agent import research
from bias_agent import run_bias_check
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="../frontend", html=True), name="frontend")


class ResearchRequest(BaseModel):
    question: str
    bias_preference: str = "NEUTRAL"


@app.post("/research")
async def run_research(req: ResearchRequest):
    report, sources = research(req.question, req.bias_preference)

    bias_results = None
    if sources:
        bias_results = run_bias_check(req.question, sources)

    return {
        "report": report,
        "sources": sources,
        "bias_results": bias_results,
        "is_controversial": bias_results is not None
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)