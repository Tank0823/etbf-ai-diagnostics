"""
Data Ingestion and Harmonization Layer
Author: Angel B. Yanes
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModalityData:
    """Container for processed data from a single modality."""
    modality: str
    raw_data: Any
    processed_data: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)
    status: str = "pending"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DataIngestionPipeline:
    """
    Main pipeline that ingests, validates, and harmonizes data from all five modalities.
    
    Supported modalities:
    - radiology: MRI, CT, X-ray (DICOM files)
    - ultrasound: Ultrasound videos or DICOM loops
    - electrophoresis: Gel images (TIFF, PNG)
    - microscopy: Whole Slide Images (SVS, NDPI)
    - laboratory: Structured lab data (CBC, BMP, CMP, ABG) in JSON/HL7
    """
    
    def __init__(self, config: Dict[str, Any], modalities: List[str]):
        self.config = config
        self.modalities = modalities
        self._validate_modalities()
        logger.info(f"Data Ingestion Pipeline initialized for: {modalities}")
    
    def _validate_modalities(self):
        """Ensure all requested modalities are supported."""
        supported = {"radiology", "ultrasound", "electrophoresis", "microscopy", "laboratory"}
        for m in self.modalities:
            if m not in supported:
                raise ValueError(f"Unsupported modality: {m}. Supported: {supported}")
    
    async def process(self, patient_id: str, modalities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process each provided modality and return harmonized data.
        
        Args:
            patient_id: Unique patient identifier
            modalities: Dict mapping modality name to raw data (file path, bytes, etc.)
        
        Returns:
            Dict containing processed results for each modality and overall metadata.
        """
        results = {}
        available = []
        errors = []
        
        for name, data in modalities.items():
            if name not in self.modalities:
                errors.append(f"Modality '{name}' not in pipeline configuration")
                continue
            
            try:
                # Simulate processing (in real system, this would call specialized handlers)
                processed = self._process_modality(name, data)
                results[name] = processed
                available.append(name)
                logger.info(f"Processed {name} for patient {patient_id}")
            except Exception as e:
                errors.append(f"Failed to process {name}: {str(e)}")
                logger.error(f"Error processing {name}: {e}")
                results[name] = ModalityData(
                    modality=name,
                    raw_data=data,
                    status="error",
                    metadata={"error": str(e)}
                ).__dict__
        
        return {
            "patient_id": patient_id,
            "modalities": results,
            "available_modalities": available,
            "errors": errors if errors else None,
            "timestamp": datetime.now().isoformat(),
            "total_modalities": len(modalities),
            "successful": len(available)
        }
    
    def _process_modality(self, modality: str, data: Any) -> Dict:
        """
        Placeholder for actual modality-specific processing.
        In the real system, this would route to specialized preprocessors.
        """
        # Simulate processing by returning a structured summary
        return {
            "modality": modality,
            "status": "processed",
            "data_summary": f"Received {modality} data of type {type(data).__name__}",
            "metadata": {
                "size": len(str(data)) if data else 0,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    async def add_reference_case(self, case: Dict) -> bool:
        """
        Add a new reference case to the system (for RAG).
        This will be expanded later.
        """
        logger.info(f"Reference case added (stub): {case.get('id', 'unknown')}")
        return True
