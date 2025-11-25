"""
Hierarchical DRL Coordinator
Orchestrates strategic, tactical, and operational decision making
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dqn_strategic import EnhancedDQNetwork
from models.ppo_tactical import PPOActorCritic, decode_action
from models.lstm_operational import LSTMPredictor
from preprocessing.feature_engineering import (
    create_enhanced_strategic_state,
    create_enhanced_tactical_state
)


class HierarchicalCoordinator:
    """
    Coordinates hierarchical decision-making across three DRL layers:
    1. Strategic Layer (DQN): Cloud provider selection
    2. Tactical Layer (PPO): Regional placement and memory allocation
    3. Operational Layer (LSTM): Workload prediction and resource scaling
    """

    def __init__(
        self,
        strategic_model_path: str,
        tactical_model_path: str,
        operational_model_path: str,
        device: Optional[str] = None
    ):
        """
        Initialize hierarchical coordinator

        Args:
            strategic_model_path: Path to DQN model weights
            tactical_model_path: Path to PPO model weights
            operational_model_path: Path to LSTM model weights
            device: Device to run models on ('cpu', 'cuda', or None for auto)
        """
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # Load models
        self.strategic_model = self._load_strategic_model(strategic_model_path)
        self.tactical_model = self._load_tactical_model(tactical_model_path)
        self.operational_model = self._load_operational_model(operational_model_path)

        # Set models to evaluation mode
        self.strategic_model.eval()
        self.tactical_model.eval()
        self.operational_model.eval()

        # Cloud provider mapping
        self.cloud_providers = ['AWS', 'Azure', 'GCP']

        print(f"Hierarchical Coordinator initialized on device: {self.device}")
        print(f"  ✓ Strategic model loaded")
        print(f"  ✓ Tactical model loaded")
        print(f"  ✓ Operational model loaded")

    def _load_strategic_model(self, model_path: str) -> EnhancedDQNetwork:
        """Load strategic DQN model"""
        model = EnhancedDQNetwork(state_size=14, action_size=3)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model

    def _load_tactical_model(self, model_path: str) -> PPOActorCritic:
        """Load tactical PPO model"""
        model = PPOActorCritic(state_dim=11, action_dim=24)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model

    def _load_operational_model(self, model_path: str) -> LSTMPredictor:
        """Load operational LSTM model"""
        model = LSTMPredictor(input_dim=5, hidden_dim1=128, hidden_dim2=64, output_dim=3, dropout=0.2)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model

    def make_decision(
        self,
        strategic_state: np.ndarray,
        tactical_state: np.ndarray,
        operational_sequence: Optional[np.ndarray],
        app_profile: Dict
    ) -> Dict:
        """
        Make hierarchical decision across all three layers

        Args:
            strategic_state: Strategic features (10-dim)
            tactical_state: Tactical features (7-dim)
            operational_sequence: Operational sequence (12, 5) - optional
            app_profile: Application profile dict

        Returns:
            decision: Dict with keys:
                - cloud_provider: str (AWS/Azure/GCP)
                - region: str (e.g., 'us-east-1')
                - memory_mb: int (e.g., 512)
                - predicted_resources: Dict (if operational_sequence provided)
        """
        # Layer 1: Strategic - Cloud provider selection
        cloud_idx, cloud_q_values = self._strategic_decision(strategic_state, app_profile)
        cloud_provider = self.cloud_providers[cloud_idx]

        # Layer 2: Tactical - Regional placement and memory allocation
        region, memory_mb, tactical_probs = self._tactical_decision(
            tactical_state, cloud_idx
        )

        # Layer 3: Operational - Workload prediction (if sequence available)
        predicted_resources = None
        if operational_sequence is not None:
            predicted_resources = self._operational_prediction(operational_sequence)

        decision = {
            'cloud_provider': cloud_provider,
            'cloud_provider_idx': cloud_idx,
            'cloud_q_values': cloud_q_values.tolist(),
            'region': region,
            'memory_mb': memory_mb,
            'tactical_action_probs': tactical_probs.tolist(),
            'predicted_resources': predicted_resources
        }

        return decision

    def _strategic_decision(
        self,
        strategic_state: np.ndarray,
        app_profile: Dict
    ) -> Tuple[int, np.ndarray]:
        """
        Strategic layer: Select cloud provider

        Args:
            strategic_state: Strategic features (10-dim)
            app_profile: Application profile dict

        Returns:
            cloud_idx: Selected cloud provider index (0=AWS, 1=Azure, 2=GCP)
            q_values: Q-values for all cloud providers
        """
        # Create enhanced state with app context
        enhanced_state = create_enhanced_strategic_state(strategic_state, app_profile)

        # Convert to tensor
        state_tensor = torch.FloatTensor(enhanced_state).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            q_values = self.strategic_model(state_tensor)

        # Select action with highest Q-value
        cloud_idx = q_values.argmax(dim=1).item()
        q_values_np = q_values.cpu().numpy().squeeze()

        return cloud_idx, q_values_np

    def _tactical_decision(
        self,
        tactical_state: np.ndarray,
        strategic_cloud: int
    ) -> Tuple[str, int, np.ndarray]:
        """
        Tactical layer: Select region and memory allocation

        Args:
            tactical_state: Tactical features (7-dim)
            strategic_cloud: Cloud provider from strategic layer

        Returns:
            region: Selected region (e.g., 'us-east-1')
            memory_mb: Selected memory allocation (MB)
            action_probs: Action probability distribution
        """
        # Create enhanced state with strategic context
        enhanced_state = create_enhanced_tactical_state(tactical_state, strategic_cloud)

        # Convert to tensor
        state_tensor = torch.FloatTensor(enhanced_state).unsqueeze(0).to(self.device)

        # Forward pass - use deterministic action
        with torch.no_grad():
            action_tensor, _, _ = self.tactical_model.act(state_tensor, deterministic=True)
            # Also get action probabilities
            action_logits, _ = self.tactical_model(state_tensor)
            action_probs = torch.softmax(action_logits, dim=-1).cpu().numpy().squeeze()

        action = action_tensor.item()

        # Decode action to region and memory
        region, memory_mb = decode_action(action)

        return region, memory_mb, action_probs

    def _operational_prediction(
        self,
        operational_sequence: np.ndarray
    ) -> Dict:
        """
        Operational layer: Predict workload and resource requirements

        Args:
            operational_sequence: Sequence of operational features (12, 5)

        Returns:
            predictions: Dict with predicted resources:
                - request_rate: Predicted invocation rate
                - memory_util: Predicted memory utilization
                - cpu_util: Predicted CPU utilization
        """
        # Convert to tensor
        if operational_sequence.ndim == 2:
            # Add batch dimension
            operational_sequence = operational_sequence[np.newaxis, :, :]

        sequence_tensor = torch.FloatTensor(operational_sequence).to(self.device)

        # Forward pass
        with torch.no_grad():
            predictions = self.operational_model(sequence_tensor)

        predictions_np = predictions.cpu().numpy().squeeze()

        return {
            'request_rate': float(predictions_np[0]),
            'memory_util': float(predictions_np[1]),
            'cpu_util': float(predictions_np[2])
        }


if __name__ == "__main__":
    print("Testing Hierarchical Coordinator...")

    # This is just a structure test - actual model loading requires trained models
    # In production, you would initialize like this:
    #
    # coordinator = HierarchicalCoordinator(
    #     strategic_model_path='path/to/best_enhanced_dqn.pt',
    #     tactical_model_path='path/to/best_ppo_tactical.pt',
    #     operational_model_path='path/to/best_lstm_predictor.pt'
    # )
    #
    # decision = coordinator.make_decision(
    #     strategic_state=np.random.rand(10),
    #     tactical_state=np.random.rand(7),
    #     operational_sequence=np.random.rand(12, 5),
    #     app_profile={'cold_start_rate': 0.1, 'sla_violation_rate': 0.05, ...}
    # )

    print("✓ Hierarchical Coordinator module structure verified")
