"""
DQN Strategic Layer Model
Enhanced Deep Q-Network for cloud provider selection in multi-cloud serverless orchestration.

This module contains the EnhancedDQNetwork class that performs strategic-level decisions
for selecting cloud providers (AWS, Azure, GCP) based on application context and workload patterns.
"""

import torch
import torch.nn as nn
import numpy as np


class EnhancedDQNetwork(nn.Module):
    """
    Enhanced DQN with separate encoders for strategic and application context.

    This network architecture is designed for strategic cloud provider selection,
    processing both strategic features (workload patterns, temporal features) and
    application-specific context (cold start rates, SLA requirements, burst patterns).

    Architecture:
        Strategic Encoder: 10 -> 64 (processes temporal and workload features)
        App Context Encoder: 4 -> 32 (processes application-specific features)
        Fusion Network: 96 -> 128 -> 64 -> 3 (combines encoders and produces Q-values)

    Args:
        state_size (int): Total input state dimension (default: 14)
            - First 10 features: strategic state (hour, day_of_week, is_weekend,
              is_business_hours, invocation_rate, is_bursty, avg_duration,
              avg_cost, avg_carbon, memory_mb)
            - Last 4 features: application context (cold_start_rate, sla_violation_rate,
              avg_invocation_rate, workload_type)
        action_size (int): Number of cloud provider actions (default: 3 for AWS, Azure, GCP)
        hidden_sizes (list): Hidden layer sizes for encoders and fusion network
            [strategic_encoder_size, app_context_encoder_size, fusion_hidden_size]

    Example:
        >>> model = EnhancedDQNetwork(state_size=14, action_size=3)
        >>> state = torch.randn(32, 14)  # batch of 32 states
        >>> q_values = model(state)  # shape: (32, 3)
    """

    def __init__(self, state_size=14, action_size=3, hidden_sizes=[64, 32, 128]):
        super(EnhancedDQNetwork, self).__init__()

        # Strategic encoder (10 -> 64)
        # Processes temporal and workload features
        self.strategic_encoder = nn.Sequential(
            nn.Linear(10, hidden_sizes[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_sizes[0]),
            nn.Dropout(0.1)
        )

        # App context encoder (4 -> 32)
        # Processes application-specific features
        self.app_context_encoder = nn.Sequential(
            nn.Linear(4, hidden_sizes[1]),
            nn.ReLU(),
            nn.LayerNorm(hidden_sizes[1]),
            nn.Dropout(0.1)
        )

        # Fusion network (96 -> 3)
        # Combines both encoders to produce Q-values for each cloud provider
        fusion_input_size = hidden_sizes[0] + hidden_sizes[1]
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_sizes[2]),
            nn.ReLU(),
            nn.LayerNorm(hidden_sizes[2]),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[2], 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming normal initialization for ReLU activations.
        Biases are initialized with small positive values to avoid dead neurons.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): Input state tensor of shape (batch_size, state_size)
                First 10 dimensions: strategic features
                Last 4 dimensions: application context features

        Returns:
            torch.Tensor: Q-values for each action of shape (batch_size, action_size)
        """
        # Split state into strategic and application context components
        strategic = state[:, :10]
        app_context = state[:, 10:]

        # Encode strategic features
        strategic_emb = self.strategic_encoder(strategic)

        # Encode application context
        app_emb = self.app_context_encoder(app_context)

        # Combine embeddings
        combined = torch.cat([strategic_emb, app_emb], dim=1)

        # Generate Q-values
        q_values = self.fusion(combined)

        return q_values


def create_dqn_model(state_size=14, action_size=3, device='cpu'):
    """
    Factory function to create and initialize an EnhancedDQNetwork.

    Args:
        state_size (int): Input state dimension
        action_size (int): Number of actions (cloud providers)
        device (str): Device to place the model on ('cpu' or 'cuda')

    Returns:
        EnhancedDQNetwork: Initialized model on the specified device
    """
    model = EnhancedDQNetwork(state_size=state_size, action_size=action_size)
    model = model.to(device)
    return model


def load_dqn_model(checkpoint_path, state_size=14, action_size=3, device='cpu'):
    """
    Load a trained EnhancedDQNetwork from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint (.pt file)
        state_size (int): Input state dimension
        action_size (int): Number of actions
        device (str): Device to place the model on

    Returns:
        EnhancedDQNetwork: Loaded model in evaluation mode
    """
    model = create_dqn_model(state_size, action_size, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing EnhancedDQNetwork...")

    # Create model
    model = EnhancedDQNetwork(state_size=14, action_size=3)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    batch_size = 32
    test_state = torch.randn(batch_size, 14)
    q_values = model(test_state)

    print(f"Input shape: {test_state.shape}")
    print(f"Output shape: {q_values.shape}")
    print(f"Sample Q-values: {q_values[0].detach().numpy()}")
    print("\nModel test successful!")
