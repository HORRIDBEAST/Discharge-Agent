"""
FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.utils.logger import configure_logging
from app.configs.settings import get_settings

configure_logging()
settings = get_settings()

app = FastAPI(
    title="Discharge Summary Agent",
    description="Agentic AI system for clinical discharge summary generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "Discharge Summary Agent",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
