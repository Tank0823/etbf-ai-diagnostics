"""
Explainability Engine
Author: Angel B. Yanes

This module provides multi-level explainability:
- Visual explainability (Grad-CAM heatmaps - simulated)
- Feature-level explainability (SHAP values - simulated)
- Reasoning explainability (Chain-of-Thought traces)
- Concept-based explanations
"""

import logging
import base64
import io
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Container for a complete explanation."""
    visual: Dict[str, Any] = field(default_factory=dict)
    feature_level: Dict[str, Any] = field(default_factory=dict)
    reasoning: Dict[str, Any] = field(default_factory=dict)
    concept_based: Dict[str, Any] = field(default_factory=dict)


class ExplainabilityEngine:
    """
    Multi-level explainability engine for the eTBF system.
    
    Generates visual, feature-level, reasoning-based, and concept-based
    explanations for diagnostic outputs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.grad_cam_enabled = config.get("grad_cam_enabled", True)
        self.shap_enabled = config.get("shap_enabled", True)
        self.attention_visualization = config.get("attention_visualization", True)
        self.concept_based = config.get("concept_based_explanations", True)
        self.concept_vocabulary = config.get("concept_vocabulary", [
            "nuclear_atypia", "mitotic_figures", "glandular_formation",
            "stromal_invasion", "tumor_density", "band_intensity",
            "monoclonal_spike", "calcification", "necrosis",
            "infection_risk", "renal_impairment", "hepatic_dysfunction",
            "acid_base_imbalance", "inflammatory_response", "anemia"
        ])
        
        logger.info("Explainability Engine initialized")
    
    async def generate(
        self,
        diagnosis: Dict[str, Any],
        features: Dict[str, Any],
        model_activations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate multi-level explanations for a diagnosis.
        
        Args:
            diagnosis: Diagnosis result from fusion engine
            features: Extracted features from MoE encoders
            model_activations: Activations from the model
        
        Returns:
            Dictionary containing all explanation types
        """
        explanations = {
            "visual": {},
            "feature_level": {},
            "reasoning": {},
            "concept_based": {}
        }
        
        # Visual explanations (simulated Grad-CAM)
        if self.grad_cam_enabled:
            explanations["visual"]["grad_cam"] = self._simulate_grad_cam(features)
        
        # Feature-level explanations (simulated SHAP)
        if self.shap_enabled:
            explanations["feature_level"]["shap"] = self._simulate_shap(features, diagnosis)
        
        # Reasoning explanations
        explanations["reasoning"]["chain_of_thought"] = diagnosis.get("reasoning_trace", "")
        explanations["reasoning"]["supporting_features"] = diagnosis.get("supporting_features", [])
        explanations["reasoning"]["refuting_features"] = diagnosis.get("refuting_features", [])
        
        # Concept-based explanations
        if self.concept_based:
            explanations["concept_based"] = self._simulate_concept_explanations(features, diagnosis)
        
        return explanations
    
    def _simulate_grad_cam(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate Grad-CAM heatmaps for each modality.
        
        In a real system, this would use gradient backpropagation.
        """
        result = {}
        modalities = features.get("modality_embeddings", {}).keys()
        
        for modality in modalities:
            # Simulate a heatmap as a base64-encoded PNG (for demo, we'll create a tiny image)
            # We'll generate a small random heatmap pattern
            import matplotlib.pyplot as plt
            import numpy as np
            from io import BytesIO
            
            # Create a random heatmap (for demonstration)
            heatmap = np.random.rand(64, 64)
            # Add a "hot spot" to simulate attention
            cx, cy = random.randint(10, 54), random.randint(10, 54)
            for i in range(64):
                for j in range(64):
                    dist = np.sqrt((i-cx)**2 + (j-cy)**2)
                    heatmap[i, j] += np.exp(-dist/15)
            # Normalize
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(heatmap, cmap='jet', alpha=0.7)
            ax.axis('off')
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            result[modality] = {
                "image_data": img_base64,
                "heatmap": heatmap.tolist(),
                "modality": modality
            }
        
        return result
    
    def _simulate_shap(self, features: Dict[str, Any], diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate SHAP values for feature importance.
        
        In a real system, this would compute actual SHAP values.
        """
        shap_results = {}
        clinical_features = features.get("clinical_features", {})
        
        for modality, feat in clinical_features.items():
            if isinstance(feat, dict):
                # Generate random SHAP values for features
                feature_names = list(feat.keys())
                shap_values = [random.uniform(-0.3, 0.3) for _ in feature_names]
                # Ensure at least one positive and one negative
                if all(v >= 0 for v in shap_values):
                    shap_values[0] = -0.1
                if all(v <= 0 for v in shap_values):
                    shap_values[0] = 0.1
                
                shap_results[modality] = {
                    "feature_names": feature_names,
                    "shap_values": shap_values,
                    "base_value": 0.5
                }
        
        return shap_results
    
    def _simulate_concept_explanations(self, features: Dict[str, Any], diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate concept-based explanations.
        
        Maps features to clinical concepts to provide interpretable explanations.
        """
        concept_explanations = {}
        
        # Extract clinical features from modalities
        clinical_features = features.get("clinical_features", {})
        all_feature_values = {}
        for modality, feat in clinical_features.items():
            if isinstance(feat, dict):
                all_feature_values.update(feat)
        
        for concept in self.concept_vocabulary:
            # Check if concept is present in any feature or diagnosis
            concept_lower = concept.lower()
            present = False
            
            # Check in features
            for key, value in all_feature_values.items():
                if concept_lower in str(key).lower() or concept_lower in str(value).lower():
                    present = True
                    break
            
            # Check in diagnosis
            if not present:
                if concept_lower in diagnosis.get("primary", "").lower():
                    present = True
            
            concept_explanations[concept] = {
                "present": present,
                "in_diagnosis": concept_lower in diagnosis.get("primary", "").lower(),
                "in_features": any(concept_lower in str(v).lower() for v in all_feature_values.values()),
                "description": self._get_concept_description(concept)
            }
        
        return concept_explanations
    
    def _get_concept_description(self, concept: str) -> str:
        """Get a description of a clinical concept."""
        descriptions = {
            "nuclear_atypia": "Abnormal nuclear morphology indicating potential malignancy",
            "mitotic_figures": "Cell division observed, indicating rapid cell proliferation",
            "glandular_formation": "Tissue architecture showing gland-like structures",
            "stromal_invasion": "Tumor cells invading surrounding tissue",
            "tumor_density": "Radiological density of abnormal tissue",
            "band_intensity": "Intensity of protein bands on electrophoresis",
            "monoclonal_spike": "Abnormal protein band indicating monoclonal gammopathy",
            "calcification": "Calcium deposits visible on imaging",
            "necrosis": "Dead tissue, often indicating aggressive tumor growth",
            "infection_risk": "Elevated infection markers",
            "renal_impairment": "Abnormal kidney function markers",
            "hepatic_dysfunction": "Abnormal liver enzyme levels",
            "acid_base_imbalance": "pH or bicarbonate abnormalities",
            "inflammatory_response": "Elevated inflammation markers",
            "anemia": "Low hemoglobin or red blood cell count"
        }
        return descriptions.get(concept, "Clinical concept")
