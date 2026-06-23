"""
API Routes for the eTBF System
Author: Angel B. Yanes
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse

# Import core components
from app.main import get_components

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "operational",
        "version": "1.0.0",
        "modalities": ["radiology", "ultrasound", "electrophoresis", "microscopy", "laboratory"]
    }


@router.post("/diagnose")
async def diagnose(
    patient_id: str = Form(...),
    radiology: Optional[UploadFile] = None,
    ultrasound: Optional[UploadFile] = None,
    electrophoresis: Optional[UploadFile] = None,
    microscopy: Optional[UploadFile] = None,
    laboratory: Optional[UploadFile] = None
):
    """
    Perform a multimodal diagnosis using all five modalities.
    
    This endpoint:
    1. Ingests and processes all provided modalities
    2. Encodes each modality through its expert encoder
    3. Retrieves similar cases from the vector database
    4. Fuses all information and generates a diagnosis
    5. Produces explainability outputs
    """
    try:
        # Get core components
        components = get_components()
        
        # Save uploaded files
        modalities = {}
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        for name, file in [
            ("radiology", radiology),
            ("ultrasound", ultrasound),
            ("electrophoresis", electrophoresis),
            ("microscopy", microscopy),
            ("laboratory", laboratory)
        ]:
            if file and file.filename:
                modality_dir = upload_dir / name
                modality_dir.mkdir(parents=True, exist_ok=True)
                file_path = modality_dir / file.filename
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                modalities[name] = str(file_path)
                logger.info(f"Saved {name} file: {file_path}")
        
        if not modalities:
            raise HTTPException(400, "At least one modality is required")
        
        # Step 1: Data Ingestion and Harmonization
        logger.info(f"Processing patient {patient_id} with modalities: {list(modalities.keys())}")
        harmonized_data = await components["data_ingestion"].process(
            patient_id=patient_id,
            modalities=modalities
        )
        
        # Step 2: Mixture of Experts Encoding
        encoded_features = await components["moe_encoder"].encode(harmonized_data)
        
        # Step 3: Retrieval-Augmented Generation
        similar_cases = await components["rag_engine"].retrieve(
            query_embeddings=encoded_features["multimodal_embedding"],
            top_k=5
        )
        
        # Step 4: Multimodal Fusion and Reasoning
        diagnosis = await components["fusion_engine"].generate_diagnosis(
            features=encoded_features,
            similar_cases=similar_cases,
            patient_metadata=harmonized_data.get("metadata", {})
        )
        
        # Step 5: Explainability
        explanations = await components["explainability"].generate(
            diagnosis=diagnosis,
            features=encoded_features,
            model_activations=components["fusion_engine"].get_activations()
        )
        
        # Assemble result
        result = {
            "patient_id": patient_id,
            "diagnosis": diagnosis,
            "differential_diagnosis": diagnosis.get("differential", []),
            "reasoning": diagnosis.get("reasoning_trace", ""),
            "temporal_context": diagnosis.get("temporal_context", {}),
            "confidence": diagnosis.get("confidence_score", diagnosis.get("confidence", 0.0)),
            "evidence": {
                "similar_cases": similar_cases,
                "supporting_features": diagnosis.get("supporting_features", [])
            },
            "explainability": explanations,
            "metadata": {
                "modalities_processed": list(modalities.keys()),
                "modalities_available": harmonized_data.get("available_modalities", []),
                "timestamp": harmonized_data.get("timestamp", "")
            },
            "status": "success"
        }
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Diagnosis failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Diagnosis failed: {str(e)}")


@router.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        components = get_components()
        rag_stats = await components["rag_engine"].get_collection_stats()
        fl_status = await components["fl_coordinator"].get_status()
        
        return {
            "vector_db": rag_stats,
            "modalities": ["radiology", "ultrasound", "electrophoresis", "microscopy", "laboratory"],
            "federated_learning": fl_status,
            "version": "1.0.0",
            "status": "operational"
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "version": "1.0.0"
        }


@router.post("/reference")
async def add_reference(
    case_id: str = Form(...),
    diagnosis: str = Form(...),
    outcome: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Add a reference case to the vector database.
    """
    try:
        components = get_components()
        rag_engine = components["rag_engine"]
        
        # Generate a simulated embedding (in production, this would come from the encoders)
        import random
        embedding = [random.uniform(-0.1, 0.1) for _ in range(384)]
        
        await rag_engine.add_reference_case(
            case_id=case_id,
            embedding=embedding,
            metadata={
                "diagnosis": diagnosis,
                "outcome": outcome,
                "files": [f.filename for f in files],
                "timestamp": "2026-06-23T00:00:00"
            }
        )
        
        return {
            "status": "success",
            "case_id": case_id,
            "message": f"Reference case {case_id} added successfully"
        }
        
    except Exception as e:
        logger.error(f"Add reference failed: {e}")
        raise HTTPException(500, str(e))


@router.get("/reference/{case_id}")
async def get_reference(case_id: str):
    """Get a reference case from the database."""
    try:
        components = get_components()
        case = await components["rag_engine"].get_reference_case(case_id)
        if not case:
            raise HTTPException(404, "Case not found")
        return case
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get reference failed: {e}")
        raise HTTPException(500, str(e))


@router.delete("/reference/{case_id}")
async def delete_reference(case_id: str):
    """Delete a reference case from the database."""
    try:
        components = get_components()
        success = await components["rag_engine"].delete_reference_case(case_id)
        if not success:
            raise HTTPException(404, "Case not found")
        return {"status": "deleted", "case_id": case_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete reference failed: {e}")
        raise HTTPException(500, str(e))


@router.post("/fl/train")
async def federated_learn(
    client_id: str = Form(...),
    data_size: int = Form(...)
):
    """
    Simulate federated learning training.
    """
    try:
        components = get_components()
        fl_coordinator = components["fl_coordinator"]
        
        result = await fl_coordinator.simulate_training(
            client_id=client_id,
            data_size=data_size
        )
        
        return {
            "status": "success",
            "client_id": client_id,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"FL training failed: {e}")
        raise HTTPException(500, str(e))


@router.get("/fl/status")
async def fl_status():
    """Get federated learning status."""
    try:
        components = get_components()
        status = await components["fl_coordinator"].get_status()
        return status
    except Exception as e:
        logger.error(f"FL status failed: {e}")
        raise HTTPException(500, str(e))
