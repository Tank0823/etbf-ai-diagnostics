"""
eTBF AI Diagnostics System - Main Application Entry Point
Author: Angel B. Yanes
"""

import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings

# Import core modules
from app.core.data_ingestion import DataIngestionPipeline
from app.core.moe_encoders import MixtureOfExpertsEncoder
from app.core.rag_engine import RAGEngine
from app.core.fusion_engine import MultimodalFusionEngine
from app.core.explainability import ExplainabilityEngine
from app.core.federated_learning import FederatedLearningCoordinator
from app.api.routes import router as api_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="eTBF AI Diagnostics System",
    description="Enhanced Total Biological Fingerprint AI Diagnostics System",
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

# Include API router
app.include_router(api_router, prefix="/api")

# Setup templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Initialize core components (lazy initialization)
_components = {}


def get_components():
    """Get or initialize core components."""
    global _components
    if not _components:
        settings = get_settings()
        _components = {
            "data_ingestion": DataIngestionPipeline(
                config=settings.data_ingestion.dict(),
                modalities=settings.supported_modalities
            ),
            "moe_encoder": MixtureOfExpertsEncoder(
                config=settings.moe_config.dict()
            ),
            "rag_engine": RAGEngine(
                config=settings.rag_config.dict(),
                vector_db_path=settings.vector_db_path
            ),
            "fusion_engine": MultimodalFusionEngine(
                config=settings.fusion_config.dict(),
                llm_model_name=settings.llm_model_name
            ),
            "explainability": ExplainabilityEngine(
                config=settings.explainability_config.dict()
            ),
            "fl_coordinator": FederatedLearningCoordinator(
                config=settings.fl_config.dict()
            )
        }
        logger.info("All components initialized")
    return _components


# ============================================
# UI Routes
# ============================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "app_name": "eTBF AI Diagnostics System"}
    )


@app.get("/diagnose", response_class=HTMLResponse)
async def diagnose_page(request: Request):
    """Diagnosis form page."""
    settings = get_settings()
    return templates.TemplateResponse(
        "diagnose.html",
        {"request": request, "modalities": settings.supported_modalities}
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    """Administration page."""
    return templates.TemplateResponse(
        "admin.html",
        {"request": request}
    )


# ============================================
# Demo Diagnosis Endpoint (for testing)
# ============================================

@app.post("/api/demo/diagnose")
async def demo_diagnose(patient_id: str):
    """
    Demo endpoint that uses all core components.
    
    This demonstrates the full eTBF pipeline:
    1. Data Ingestion (simulated)
    2. MoE Encoding
    3. RAG Retrieval
    4. Fusion & Diagnosis
    5. Explainability
    """
    try:
        components = get_components()
        
        # 1. Simulate data ingestion
        demo_modalities = {
            "radiology": {"file": "demo.dcm", "modality": "CT"},
            "laboratory": {"results": [{"test": "WBC", "value": 8.5}, {"test": "Hemoglobin", "value": 14.2}]}
        }
        
        harmonized = await components["data_ingestion"].process(
            patient_id=patient_id,
            modalities=demo_modalities
        )
        
        # 2. MoE Encoding
        encoded = await components["moe_encoder"].encode(harmonized)
        
        # 3. RAG Retrieval
        similar_cases = await components["rag_engine"].retrieve(
            query_embeddings=encoded["multimodal_embedding"],
            top_k=3
        )
        
        # 4. Fusion & Diagnosis
        diagnosis = await components["fusion_engine"].generate_diagnosis(
            features=encoded,
            similar_cases=similar_cases,
            patient_metadata={"patient_id": patient_id}
        )
        
        # 5. Explainability
        explanations = await components["explainability"].generate(
            diagnosis=diagnosis,
            features=encoded,
            model_activations=components["fusion_engine"].get_activations()
        )
        
        return JSONResponse({
            "patient_id": patient_id,
            "diagnosis": diagnosis,
            "similar_cases": similar_cases,
            "explanations": explanations,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Demo diagnosis failed: {e}")
        return JSONResponse({
            "patient_id": patient_id,
            "status": "error",
            "message": str(e)
        }, status_code=500)


# ============================================
# Health Check
# ============================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "operational",
        "version": "1.0.0",
        "modalities": ["radiology", "ultrasound", "electrophoresis", "microscopy", "laboratory"],
        "components": list(get_components().keys())
    }
