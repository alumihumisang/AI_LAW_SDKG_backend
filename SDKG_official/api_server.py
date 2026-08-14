from __future__ import annotations

"""
FastAPI entry point for the SDKG indictment generator.

Run from the repository root:
    uvicorn SDKG_official.api_server:app --host 0.0.0.0 --port 8000
"""

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .web_indictment_generator import (
    DEFAULT_EXPERIMENT,
    DEFAULT_TOP_K,
    GenerationConfig,
    IndictmentGenerator,
    analyze_query_text,
)


class GenerateRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    category: str = ""
    query_id: int | str = "web"
    experiment: str = DEFAULT_EXPERIMENT
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)
    model: str = "gemma3:27b"
    llm_url: str = "http://localhost:11434/api/generate"
    case_limit: int | None = Field(None, ge=1)


class AnalyzeRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    category: str = ""


app = FastAPI(title="AI_LAW SDKG Indictment API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
def generate(request: GenerateRequest) -> dict:
    try:
        generator = get_generator(
            request.experiment,
            request.top_k,
            request.model,
            request.llm_url,
            request.case_limit,
        )
        return generator.generate(
            request.query_text,
            category=request.category,
            query_id=request.query_id,
            top_k=request.top_k,
            model=request.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    try:
        return analyze_query_text(request.query_text, category=request.category)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"analysis failed: {exc}") from exc


@lru_cache(maxsize=8)
def get_generator(
    experiment: str,
    top_k: int,
    model: str,
    llm_url: str,
    case_limit: int | None,
) -> IndictmentGenerator:
    generator = IndictmentGenerator(
        GenerationConfig(
            experiment=experiment,
            top_k=top_k,
            model=model,
            llm_url=llm_url,
            case_limit=case_limit,
        )
    )
    generator.load()
    return generator
