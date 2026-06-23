"""
Laboratory Data Preprocessor - CBC, BMP, CMP, ABG
Author: Angel B. Yanes

This module handles preprocessing of structured laboratory data:
- Complete Blood Count (CBC)
- Basic Metabolic Panel (BMP)
- Comprehensive Metabolic Panel (CMP)
- Arterial Blood Gas (ABG)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class LaboratoryPreprocessor:
    """
    Preprocessor for structured laboratory data.
    
    Handles:
    - HL7/FHIR data parsing
    - Dynamic unit standardization
    - Missing data imputation
    - Temporal structuring for trend analysis
    - Feature extraction for key biomarkers
    """
    
    # Reference ranges for abnormality detection
    REFERENCE_RANGES = {
        "WBC": {"low": 4.0, "high": 11.0, "unit": "cells/uL"},
        "Hemoglobin": {"low": 12.0, "high": 16.0, "unit": "g/dL"},
        "Hematocrit": {"low": 36.0, "high": 48.0, "unit": "%"},
        "Platelets": {"low": 150, "high": 400, "unit": "cells/uL"},
        "Sodium": {"low": 135, "high": 145, "unit": "mmol/L"},
        "Potassium": {"low": 3.5, "high": 5.0, "unit": "mmol/L"},
        "Glucose": {"low": 70, "high": 100, "unit": "mg/dL"},
        "Creatinine": {"low": 0.6, "high": 1.2, "unit": "mg/dL"},
        "pH": {"low": 7.35, "high": 7.45, "unit": "pH"}
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.temporal_window = config.get("lab_temporal_window", 7)
        self.imputation_method = config.get("lab_imputation_method", "mean")
        self.standardize_units = config.get("lab_unit_standardization", True)
    
    def process(self, file_path: str) -> Dict[str, Any]:
        """
        Process structured laboratory data from file.
        
        Args:
            file_path: Path to laboratory data file (JSON, HL7, or XML)
        
        Returns:
            Dictionary containing processed laboratory data
        """
        path = Path(file_path)
        
        if path.suffix.lower() == ".json":
            return self._process_json(path)
        elif path.suffix.lower() == ".hl7":
            return self._process_hl7(path)
        else:
            # Try to parse as JSON
            try:
                return self._process_json(path)
            except:
                raise ValueError(f"Unsupported laboratory format: {file_path}")
    
    def _process_json(self, file_path: Path) -> Dict[str, Any]:
        """Process JSON format laboratory data."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return self._process_structured_data(data)
    
    def _process_hl7(self, file_path: Path) -> Dict[str, Any]:
        """Process HL7 format laboratory data."""
        with open(file_path, "r") as f:
            content = f.read()
        
        data = self._parse_hl7(content)
        return self._process_structured_data(data)
    
    def _process_structured_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process structured laboratory data dictionary.
        """
        # Extract measurements
        measurements = self._extract_measurements(data)
        
        # Structure temporally
        temporal_data = self._structure_temporally(measurements)
        
        # Impute missing values
        temporal_data = self._impute_missing(temporal_data)
        
        # Calculate trends
        trends = self._calculate_trends(temporal_data)
        
        # Detect abnormalities
        abnormalities = self._detect_abnormalities(temporal_data)
        
        # Extract key features
        features = self._extract_features(temporal_data, trends, abnormalities)
        
        return {
            "measurements": measurements,
            "temporal_data": temporal_data.to_dict() if isinstance(temporal_data, pd.DataFrame) else {},
            "trends": trends,
            "abnormalities": abnormalities,
            "features": features,
            "metadata": {
                "num_measurements": len(measurements),
                "temporal_window": self.temporal_window,
                "imputation_method": self.imputation_method
            }
        }
    
    def _extract_measurements(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract measurements from various data formats."""
        measurements = []
        
        if "results" in data:
            for result in data["results"]:
                measurements.append({
                    "test": result.get("test", "Unknown"),
                    "value": result.get("value", None),
                    "unit": result.get("unit", ""),
                    "timestamp": result.get("timestamp", datetime.now().isoformat())
                })
        elif "observations" in data:
            for obs in data["observations"]:
                measurements.append({
                    "test": obs.get("code", "Unknown"),
                    "value": obs.get("value", None),
                    "unit": obs.get("unit", ""),
                    "timestamp": obs.get("timestamp", datetime.now().isoformat())
                })
        else:
            # Attempt to find any numeric fields
            for key, value in data.items():
                if isinstance(value, (int, float)) and key not in ["id", "patient_id"]:
                    measurements.append({
                        "test": key,
                        "value": value,
                        "unit": "",
                        "timestamp": datetime.now().isoformat()
                    })
        
        return measurements
    
    def _parse_hl7(self, content: str) -> Dict[str, Any]:
        """Parse HL7 message content."""
        data = {"measurements": []}
        lines = content.split("\n")
        
        for line in lines:
            if line.startswith("OBX|"):
                segments = line.split("|")
                if len(segments) >= 7:
                    test = segments[3] if len(segments) > 3 else ""
                    value = segments[5] if len(segments) > 5 else ""
                    unit = segments[6] if len(segments) > 6 else ""
                    
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    
                    data["measurements"].append({
                        "test": test,
                        "value": value,
                        "unit": unit,
                        "timestamp": datetime.now().isoformat()
                    })
        
        return data
    
    def _structure_temporally(self, measurements: List[Dict[str, Any]]) -> pd.DataFrame:
        """Structure measurements as a time series."""
        df = pd.DataFrame(measurements)
        
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
        
        if "test" in df.columns and "value" in df.columns:
            pivot = df.pivot_table(
                index="timestamp",
                columns="test",
                values="value",
                aggfunc="first"
            )
            return pivot
        
        return df
    
    def _impute_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values."""
        if isinstance(data, pd.DataFrame) and not data.empty:
            if self.imputation_method == "mean":
                data = data.fillna(data.mean())
            else:
                # Forward fill with decay
                data = data.ffill(limit=3)
                data = data.bfill(limit=3)
                data = data.fillna(data.mean())
        
        return data
    
    def _calculate_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trends for key biomarkers."""
        trends = {}
        
        if not isinstance(data, pd.DataFrame) or data.empty:
            return trends
        
        for col in data.columns:
            values = data[col].dropna()
            if len(values) >= 2:
                x = list(range(len(values)))
                if len(x) > 1:
                    # Simple trend calculation
                    slope = (values.iloc[-1] - values.iloc[0]) / (len(values) - 1)
                    trends[col] = {
                        "slope": float(slope),
                        "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                        "magnitude": abs(float(slope))
                    }
        
        return trends
    
    def _detect_abnormalities(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect abnormalities in laboratory values."""
        abnormalities = {}
        
        if not isinstance(data, pd.DataFrame) or data.empty:
            return abnormalities
        
        for col in data.columns:
            if col in self.REFERENCE_RANGES:
                ranges = self.REFERENCE_RANGES[col]
                values = data[col].dropna()
                
                if len(values) > 0:
                    latest = values.iloc[-1]
                    low = ranges.get("low")
                    high = ranges.get("high")
                    
                    if low is not None and latest < low:
                        abnormalities[col] = {"status": "low", "value": float(latest)}
                    elif high is not None and latest > high:
                        abnormalities[col] = {"status": "high", "value": float(latest)}
                    else:
                        abnormalities[col] = {"status": "normal", "value": float(latest)}
        
        return abnormalities
    
    def _extract_features(
        self,
        data: pd.DataFrame,
        trends: Dict[str, Any],
        abnormalities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract key clinical features from laboratory data."""
        features = {}
        
        if not isinstance(data, pd.DataFrame) or data.empty:
            return features
        
        # Infection risk (WBC)
        if "WBC" in data.columns:
            wbc = data["WBC"].dropna()
            if len(wbc) > 0:
                features["infection_risk_score"] = min(wbc.iloc[-1] / 11.0, 2.0)
        
        # Renal function
        if "Creatinine" in data.columns:
            cr = data["Creatinine"].dropna()
            if len(cr) > 0:
                features["renal_function_status"] = "normal" if cr.iloc[-1] < 1.2 else "abnormal"
        
        # Acid-base status
        if "pH" in data.columns:
            ph = data["pH"].dropna()
            if len(ph) > 0:
                ph_value = ph.iloc[-1]
                if ph_value < 7.35:
                    features["acid_base_status"] = "acidosis"
                elif ph_value > 7.45:
                    features["acid_base_status"] = "alkalosis"
                else:
                    features["acid_base_status"] = "normal"
        
        # Anemia status
        if "Hemoglobin" in data.columns:
            hgb = data["Hemoglobin"].dropna()
            if len(hgb) > 0:
                features["anemia_status"] = "anemic" if hgb.iloc[-1] < 12.0 else "normal"
        
        # Inflammatory response
        if "WBC" in data.columns:
            wbc = data["WBC"].dropna()
            if len(wbc) > 0:
                features["acute_inflammatory_response"] = "elevated" if wbc.iloc[-1] > 11.0 else "normal"
        
        # Temporal context
        features["temporal_context"] = self._classify_temporal_context(data, trends)
        
        return features
    
    def _classify_temporal_context(self, data: pd.DataFrame, trends: Dict[str, Any]) -> Dict[str, Any]:
        """Classify disease state as acute, subacute, or chronic."""
        context = {
            "classification": "indeterminate",
            "evidence": [],
            "confidence": 0.3
        }
        
        if not isinstance(data, pd.DataFrame) or data.empty:
            return context
        
        acute_evidence = 0
        subacute_evidence = 0
        chronic_evidence = 0
        
        # Check WBC for acute infection
        if "WBC" in data.columns:
            wbc = data["WBC"].dropna()
            if len(wbc) >= 2:
                if wbc.iloc[-1] > 11.0 and wbc.iloc[-2] < 9.0:
                    acute_evidence += 1
                    context["evidence"].append("Rapid WBC elevation suggests acute process")
        
        # Check for chronic patterns (stable anemia)
        if "Hemoglobin" in data.columns:
            hgb = data["Hemoglobin"].dropna()
            if len(hgb) >= 3:
                if all(h < 12.0 for h in hgb.iloc[-3:]):
                    chronic_evidence += 1
                    context["evidence"].append("Stable anemia suggests chronic condition")
        
        # Check for subacute patterns (gradual changes)
        for test, trend in trends.items():
            if trend.get("direction") in ["increasing", "decreasing"]:
                subacute_evidence += 1
                context["evidence"].append(f"Gradual {test} change suggests subacute process")
        
        # Classify
        if acute_evidence >= 1:
            context["classification"] = "acute"
            context["confidence"] = min(0.7 + acute_evidence * 0.1, 0.95)
        elif subacute_evidence >= 2:
            context["classification"] = "subacute"
            context["confidence"] = min(0.6 + subacute_evidence * 0.1, 0.90)
        elif chronic_evidence >= 1:
            context["classification"] = "chronic"
            context["confidence"] = min(0.5 + chronic_evidence * 0.1, 0.85)
        
        return context
