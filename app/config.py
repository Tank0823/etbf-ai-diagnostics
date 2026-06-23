"""
Configuration Module for the eTBF System
Author: Angel B. Yanes
"""

from typing import List
from pydantic import BaseModel
from functools import lru_cache


class Settings(BaseModel):
    """Main settings for the eTBF system."""
    
    app_name: str = "eTBF AI Diagnostics System"
    app_version: str = "1.0.0"
    debug: bool = True
    device: str = "cuda"
    
    data_dir: str = "./data"
    upload_dir: str = "./data/uploads"
    vector_db_path: str = "./vector_db"
    model_cache_dir: str = "./model_cache"
    
    supported_modalities: List[str] = [
        "radiology", "ultrasound", "electrophoresis", 
        "microscopy", "laboratory"
    ]
    
    class Config:
        env_prefix = "ETBF_"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
