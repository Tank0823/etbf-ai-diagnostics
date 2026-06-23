"""
eTBF AI Diagnostics System - Main Application Entry Point
Author: Angel B. Yanes
"""

import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="eTBF AI Diagnostics System",
    description="Enhanced Total Biological Fingerprint AI Diagnostics",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


# ============================================
# UI Routes
# ============================================

@app.get("/")
async def index(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "app_name": "eTBF AI Diagnostics System"}
    )


@app.get("/diagnose")
async def diagnose_page(request: Request):
    """Diagnosis form page."""
    settings = get_settings()
    return templates.TemplateResponse(
        "diagnose.html",
        {
            "request": request, 
            "modalities": settings.supported_modalities
        }
    )


@app.get("/admin")
async def admin_page(request: Request):
    """Administration page."""
    return templates.TemplateResponse(
        "admin.html",
        {"request": request}
    )


# ============================================
# Health Check
# ============================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "operational",
        "version": "1.0.0",
        "modalities": ["radiology", "ultrasound", "electrophoresis", "microscopy", "laboratory"]
    }
