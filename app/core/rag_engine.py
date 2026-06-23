"""
Retrieval-Augmented Generation (RAG) Engine
Author: Angel B. Yanes

This module implements a vector database for semantic retrieval of similar cases.
It stores embeddings of historical reference cases and retrieves the most similar ones.
"""

import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ReferenceCase:
    """A single reference case stored in the vector database."""
    id: str
    embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class RAGEngine:
    """
    Simple in-memory vector database for semantic case retrieval.
    
    This is a placeholder for a real vector DB like Qdrant.
    It stores reference cases and performs cosine similarity search.
    """
    
    def __init__(self, config: Dict[str, Any], vector_db_path: str = "./vector_db"):
        self.config = config
        self.vector_db_path = vector_db_path
        self.collection_name = config.get("collection_name", "etbf_reference_cases")
        self.top_k = config.get("top_k", 5)
        self.embedding_dim = config.get("embedding_dim", 384)
        
        # In-memory storage
        self.cases: List[ReferenceCase] = []
        self._load_from_file()
        
        logger.info(f"RAG Engine initialized with {len(self.cases)} reference cases")
    
    def _load_from_file(self):
        """Load reference cases from a JSON file if it exists."""
        import os
        from pathlib import Path
        data_path = Path(self.vector_db_path) / f"{self.collection_name}.json"
        if data_path.exists():
            try:
                with open(data_path, "r") as f:
                    data = json.load(f)
                    for item in data:
                        self.cases.append(ReferenceCase(**item))
                logger.info(f"Loaded {len(self.cases)} cases from {data_path}")
            except Exception as e:
                logger.warning(f"Failed to load cases: {e}")
    
    def _save_to_file(self):
        """Save reference cases to a JSON file."""
        import os
        from pathlib import Path
        os.makedirs(self.vector_db_path, exist_ok=True)
        data_path = Path(self.vector_db_path) / f"{self.collection_name}.json"
        try:
            with open(data_path, "w") as f:
                json.dump([case.__dict__ for case in self.cases], f, indent=2)
            logger.info(f"Saved {len(self.cases)} cases to {data_path}")
        except Exception as e:
            logger.warning(f"Failed to save cases: {e}")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    async def add_reference_case(self, case_id: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """Add a new reference case to the database."""
        if len(embedding) != self.embedding_dim:
            # Pad or truncate to match expected dimension
            if len(embedding) < self.embedding_dim:
                embedding = embedding + [0.0] * (self.embedding_dim - len(embedding))
            else:
                embedding = embedding[:self.embedding_dim]
        
        # Check if case already exists
        for i, case in enumerate(self.cases):
            if case.id == case_id:
                self.cases[i] = ReferenceCase(id=case_id, embedding=embedding, metadata=metadata)
                self._save_to_file()
                logger.info(f"Updated reference case: {case_id}")
                return True
        
        self.cases.append(ReferenceCase(id=case_id, embedding=embedding, metadata=metadata))
        self._save_to_file()
        logger.info(f"Added reference case: {case_id}")
        return True
    
    async def retrieve(
        self,
        query_embeddings: List[float],
        top_k: Optional[int] = None,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar cases from the vector database.
        
        Args:
            query_embeddings: Query embedding vector
            top_k: Number of results to return (default: config value)
            filter_criteria: Optional filters (not implemented in this simple version)
        
        Returns:
            List of retrieved cases with metadata and similarity scores
        """
        if top_k is None:
            top_k = self.top_k
        
        if not self.cases:
            logger.warning("No reference cases available")
            return []
        
        # Ensure query embedding length matches expected dimension
        if len(query_embeddings) > self.embedding_dim:
            query_embeddings = query_embeddings[:self.embedding_dim]
        elif len(query_embeddings) < self.embedding_dim:
            query_embeddings = query_embeddings + [0.0] * (self.embedding_dim - len(query_embeddings))
        
        # Compute similarities for all cases
        results = []
        for case in self.cases:
            similarity = self._cosine_similarity(query_embeddings, case.embedding)
            results.append({
                "id": case.id,
                "similarity_score": similarity,
                "metadata": case.metadata
            })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Return top-k results
        top_results = results[:top_k]
        logger.info(f"Retrieved {len(top_results)} similar cases")
        return top_results
    
    async def get_reference_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific reference case by ID."""
        for case in self.cases:
            if case.id == case_id:
                return {
                    "id": case.id,
                    "embedding": case.embedding,
                    "metadata": case.metadata
                }
        return None
    
    async def delete_reference_case(self, case_id: str) -> bool:
        """Delete a reference case by ID."""
        for i, case in enumerate(self.cases):
            if case.id == case_id:
                del self.cases[i]
                self._save_to_file()
                logger.info(f"Deleted reference case: {case_id}")
                return True
        logger.warning(f"Case {case_id} not found for deletion")
        return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        return {
            "name": self.collection_name,
            "points_count": len(self.cases),
            "vectors_count": len(self.cases),
            "status": "ready"
        }
