"""
Federated Learning Coordinator
Author: Angel B. Yanes

This module enables privacy-preserving, collaborative model improvement
across a distributed network of institutions.
"""

import logging
import json
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModelUpdate:
    """Container for model weight updates from a client."""
    client_id: str
    weights: Dict[str, List[float]]
    num_samples: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class FederatedLearningCoordinator:
    """
    Federated Learning Coordinator for privacy-preserving collaborative learning.
    
    This class manages:
    - Local client registration
    - Secure aggregation of model updates (FedAvg)
    - Distribution of improved global models
    - Differential privacy (simulated)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.server_url = config.get("server_url", "http://localhost:8080")
        self.client_id = config.get("client_id", "client_001")
        self.rounds = config.get("rounds", 10)
        self.local_epochs = config.get("local_epochs", 5)
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.secure_aggregation = config.get("secure_aggregation", True)
        self.differential_privacy = config.get("differential_privacy", True)
        self.dp_epsilon = config.get("dp_epsilon", 1.0)
        
        # Storage for model updates
        self._updates: List[ModelUpdate] = []
        self._global_model: Dict[str, List[float]] = {}
        self._round_number = 0
        
        logger.info(f"Federated Learning Coordinator initialized (enabled: {self.enabled})")
    
    async def register_client(self, client_id: str) -> bool:
        """Register a new client for federated learning."""
        if client_id in [u.client_id for u in self._updates]:
            logger.warning(f"Client {client_id} already registered")
            return False
        
        logger.info(f"Client {client_id} registered")
        return True
    
    async def submit_update(self, client_id: str, weights: Dict[str, List[float]], num_samples: int) -> bool:
        """
        Submit model weight updates from a client.
        
        Args:
            client_id: Client identifier
            weights: Model weights (simulated as dict of lists)
            num_samples: Number of samples used for training
        """
        if not self.enabled:
            logger.warning("Federated learning is disabled")
            return False
        
        # Simulate secure aggregation (encryption)
        if self.secure_aggregation:
            # In real system, this would decrypt homomorphically
            # Here we just simulate by adding a random noise
            if self.differential_privacy:
                noise_scale = 1.0 / self.dp_epsilon
                for key in weights:
                    weights[key] = [w + random.gauss(0, noise_scale) for w in weights[key]]
        
        self._updates.append(ModelUpdate(
            client_id=client_id,
            weights=weights,
            num_samples=num_samples
        ))
        
        logger.info(f"Update received from client {client_id} with {num_samples} samples")
        
        # If we have enough updates, perform aggregation
        # For simulation, we aggregate after every 2 updates
        if len(self._updates) >= 2:
            await self._aggregate_updates()
        
        return True
    
    async def _aggregate_updates(self) -> None:
        """
        Aggregate model updates using Federated Averaging (FedAvg).
        
        This simulates the secure aggregation of multiple client updates.
        """
        if not self._updates:
            return
        
        self._round_number += 1
        logger.info(f"Aggregating updates for round {self._round_number}")
        
        # Compute total samples
        total_samples = sum(u.num_samples for u in self._updates)
        if total_samples == 0:
            return
        
        # Initialize global model with zeros
        first_update = self._updates[0]
        global_weights = {}
        for key in first_update.weights:
            global_weights[key] = [0.0] * len(first_update.weights[key])
        
        # Weighted average of updates
        for update in self._updates:
            weight = update.num_samples / total_samples
            for key in update.weights:
                for i, val in enumerate(update.weights[key]):
                    global_weights[key][i] += val * weight
        
        # Store global model
        self._global_model = global_weights
        
        # Clear updates after aggregation
        self._updates.clear()
        
        logger.info(f"Global model updated for round {self._round_number}")
    
    async def get_global_model(self) -> Dict[str, List[float]]:
        """Get the current global model weights."""
        return self._global_model
    
    async def get_status(self) -> Dict[str, Any]:
        """Get FL status information."""
        return {
            "enabled": self.enabled,
            "round": self._round_number,
            "pending_updates": len(self._updates),
            "global_model_size": len(self._global_model),
            "clients_registered": len(set(u.client_id for u in self._updates))
        }
    
    async def simulate_training(self, client_id: str, data_size: int) -> Dict[str, Any]:
        """
        Simulate local training on a client.
        
        This simulates the training process and returns model updates.
        """
        if not self.enabled:
            return {"status": "disabled"}
        
        # Simulate training by generating random weight updates
        # In real system, these would come from actual model training
        import numpy as np
        weights = {
            "layer1": np.random.randn(128, 64).flatten()[:64].tolist(),
            "layer2": np.random.randn(64, 32).flatten()[:64].tolist(),
            "layer3": np.random.randn(32, 16).flatten()[:64].tolist()
        }
        
        # Submit the update
        await self.submit_update(client_id, weights, data_size)
        
        return {
            "client_id": client_id,
            "data_size": data_size,
            "status": "trained",
            "weights_count": sum(len(v) for v in weights.values())
        }
