"""
LSTM Operational Layer Model
LSTM-based workload predictor for operational resource allocation in serverless functions.

This module contains the LSTMPredictor class that performs operational-level decisions
for real-time resource scaling based on temporal workload patterns.
"""

import torch
import torch.nn as nn
import numpy as np


class LSTMPredictor(nn.Module):
    """
    LSTM-based workload predictor for operational resource allocation.

    This network processes temporal sequences of operational metrics to predict
    future resource demands (request rate, memory utilization, CPU utilization).
    It uses a 2-layer LSTM architecture with dropout for regularization and
    asymmetric loss training to prioritize SLA compliance.

    Architecture:
        Input: (batch, seq_len, input_dim)
        LSTM Layer 1: 128 units with dropout=0.2
        LSTM Layer 2: 64 units with dropout=0.2
        Dense Layer: 32 neurons
        Output: 3 predictions (request_rate, memory_util, cpu_util)

    Args:
        input_dim (int): Number of features per time step (default: 5)
            Features: request_rate, memory_util, cpu_util, queue_depth, hour_sin
        hidden_dim1 (int): Hidden units in first LSTM layer (default: 128)
        hidden_dim2 (int): Hidden units in second LSTM layer (default: 64)
        output_dim (int): Number of prediction outputs (default: 3)
        dropout (float): Dropout probability for regularization (default: 0.2)

    Example:
        >>> model = LSTMPredictor(input_dim=5, hidden_dim1=128, hidden_dim2=64, output_dim=3)
        >>> # Input: batch_size=32, sequence_length=12, features=5
        >>> input_seq = torch.randn(32, 12, 5)
        >>> predictions = model(input_seq)  # shape: (32, 3)
        >>> print(f"Request rate: {predictions[0, 0]}, Memory: {predictions[0, 1]}, CPU: {predictions[0, 2]}")
    """

    def __init__(self, input_dim=5, hidden_dim1=128, hidden_dim2=64,
                 output_dim=3, dropout=0.2):
        super(LSTMPredictor, self).__init__()

        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        # First LSTM layer
        # Processes temporal sequences to capture workload patterns
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim1,
            num_layers=1,
            batch_first=True,
            dropout=0  # No dropout in single-layer LSTM
        )

        self.dropout1 = nn.Dropout(dropout)

        # Second LSTM layer
        # Refines temporal representations for better predictions
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            num_layers=1,
            batch_first=True,
            dropout=0
        )

        self.dropout2 = nn.Dropout(dropout)

        # Dense layers for final prediction
        # Maps LSTM hidden state to resource demand predictions
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize LSTM and linear layer weights.

        LSTM weights are initialized with:
        - Xavier uniform for input-hidden weights
        - Orthogonal for hidden-hidden weights
        - Zero for biases

        Linear layer weights are initialized with Xavier uniform.
        """
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param.data)

    def forward(self, x):
        """
        Forward pass through the LSTM predictor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
                - batch_size: Number of sequences in batch
                - seq_len: Length of temporal sequence (e.g., 12 for 3-minute lookback)
                - input_dim: Number of features per time step (5)

        Returns:
            torch.Tensor: Predictions of shape (batch_size, output_dim)
                - output_dim: 3 predictions (request_rate, memory_util, cpu_util)
                - All predictions are non-negative (ReLU applied)
        """
        # LSTM Layer 1
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)

        # LSTM Layer 2
        lstm2_out, (hidden, _) = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)

        # Use last hidden state for prediction
        last_hidden = hidden.squeeze(0)

        # Dense layers
        predictions = self.fc(last_hidden)

        # Ensure predictions are non-negative (resource demands cannot be negative)
        predictions = torch.relu(predictions)

        return predictions


class AsymmetricMSELoss(nn.Module):
    """
    Asymmetric Mean Squared Error Loss for operational resource allocation.

    This loss function penalizes under-provisioning (SLA violations) more heavily
    than over-provisioning (resource waste), reflecting the real-world cost
    asymmetry in serverless systems.

    Loss formulation:
        L_asymmetric = {
            β_under × (y_true - y_pred)²  if y_pred < y_true  (under-provisioning)
            β_over × (y_pred - y_true)²   if y_pred ≥ y_true  (over-provisioning)
        }

    Args:
        beta_under (float): Penalty multiplier for under-provisioning (default: 5.0)
            Higher values more strongly discourage under-provisioning
        beta_over (float): Penalty multiplier for over-provisioning (default: 1.0)
            Lower values allow some over-provisioning to ensure SLA compliance

    Example:
        >>> criterion = AsymmetricMSELoss(beta_under=5.0, beta_over=1.0)
        >>> y_pred = torch.tensor([[0.5, 0.3, 0.4]])
        >>> y_true = torch.tensor([[0.8, 0.8, 0.8]])
        >>> loss = criterion(y_pred, y_true)
        >>> print(f"Loss: {loss.item():.4f}")
    """

    def __init__(self, beta_under=5.0, beta_over=1.0):
        super(AsymmetricMSELoss, self).__init__()
        self.beta_under = beta_under
        self.beta_over = beta_over

    def forward(self, y_pred, y_true):
        """
        Compute asymmetric loss.

        Args:
            y_pred (torch.Tensor): Predicted values of shape (batch_size, output_dim)
            y_true (torch.Tensor): True values of shape (batch_size, output_dim)

        Returns:
            torch.Tensor: Scalar asymmetric MSE loss
        """
        # Compute squared errors
        squared_errors = (y_pred - y_true) ** 2

        # Create mask for under-provisioning (pred < true)
        under_provision_mask = (y_pred < y_true).float()

        # Apply asymmetric weights
        weighted_errors = (
            under_provision_mask * self.beta_under * squared_errors +
            (1 - under_provision_mask) * self.beta_over * squared_errors
        )

        # Mean over all elements
        loss = weighted_errors.mean()

        return loss


def create_lstm_model(input_dim=5, hidden_dim1=128, hidden_dim2=64, output_dim=3,
                     dropout=0.2, device='cpu'):
    """
    Factory function to create and initialize an LSTMPredictor.

    Args:
        input_dim (int): Number of features per time step
        hidden_dim1 (int): Hidden units in first LSTM layer
        hidden_dim2 (int): Hidden units in second LSTM layer
        output_dim (int): Number of prediction outputs
        dropout (float): Dropout probability
        device (str): Device to place the model on ('cpu' or 'cuda')

    Returns:
        LSTMPredictor: Initialized model on the specified device
    """
    model = LSTMPredictor(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        output_dim=output_dim,
        dropout=dropout
    )
    model = model.to(device)
    return model


def load_lstm_model(checkpoint_path, input_dim=5, hidden_dim1=128, hidden_dim2=64,
                   output_dim=3, dropout=0.2, device='cpu'):
    """
    Load a trained LSTMPredictor from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint (.pt file)
        input_dim (int): Number of features per time step
        hidden_dim1 (int): Hidden units in first LSTM layer
        hidden_dim2 (int): Hidden units in second LSTM layer
        output_dim (int): Number of prediction outputs
        dropout (float): Dropout probability
        device (str): Device to place the model on

    Returns:
        LSTMPredictor: Loaded model in evaluation mode
    """
    model = create_lstm_model(input_dim, hidden_dim1, hidden_dim2, output_dim, dropout, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def create_operational_features(data, max_rate=None):
    """
    Create operational features for LSTM input from raw data.

    This function normalizes raw metrics to [0, 1] range suitable for LSTM input.

    Args:
        data (dict or pd.DataFrame): Raw operational data containing:
            - invocation_rate: Requests per interval
            - memory_mb: Memory allocation in MB
            - duration: Execution duration in seconds
            - total_latency_ms: End-to-end latency in milliseconds
            - hour: Hour of day (0-23)
        max_rate (float, optional): Maximum request rate for normalization

    Returns:
        np.ndarray: Normalized features of shape (5,)
            [request_rate, memory_util, cpu_util, queue_depth, hour_sin]
    """
    # Request rate (log-normalized)
    raw_rate = data.get('invocation_rate', 0.0)
    if max_rate is None:
        max_rate = 100.0  # Default normalization
    request_rate = np.log1p(raw_rate) / np.log1p(max_rate + 1e-8)

    # Memory utilization (normalized to max 3008 MB)
    memory_util = data.get('memory_mb', 512.0) / 3008.0

    # CPU proxy (normalized duration)
    cpu_util = min(data.get('duration', 0.5), 1.0)

    # Queue depth (normalized latency)
    raw_queue = data.get('total_latency_ms', 0.0) / 1000.0
    queue_depth = min(raw_queue / 10.0, 1.0)  # Normalize to typical max

    # Temporal encoding (cyclical hour)
    hour = data.get('hour', 0)
    hour_sin = np.sin(2 * np.pi * hour / 24.0)

    return np.array([request_rate, memory_util, cpu_util, queue_depth, hour_sin],
                   dtype=np.float32)


if __name__ == "__main__":
    # Test the model
    print("Testing LSTMPredictor...")

    # Create model
    model = LSTMPredictor(input_dim=5, hidden_dim1=128, hidden_dim2=64, output_dim=3)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    batch_size = 32
    seq_len = 12
    input_dim = 5

    test_input = torch.randn(batch_size, seq_len, input_dim)
    predictions = model(test_input)

    print(f"\nForward pass:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions[0].detach().numpy()}")

    # Test asymmetric loss
    print("\nTesting AsymmetricMSELoss...")
    criterion = AsymmetricMSELoss(beta_under=5.0, beta_over=1.0)

    y_pred_under = torch.tensor([[0.5, 0.3, 0.4]])  # Under-provisioning
    y_pred_over = torch.tensor([[0.9, 0.9, 0.9]])   # Over-provisioning
    y_true = torch.tensor([[0.8, 0.8, 0.8]])

    loss_under = criterion(y_pred_under, y_true)
    loss_over = criterion(y_pred_over, y_true)

    print(f"  Under-provisioning loss: {loss_under.item():.6f}")
    print(f"  Over-provisioning loss: {loss_over.item():.6f}")
    print(f"  Ratio: {loss_under.item() / loss_over.item():.2f}x")

    # Test feature creation
    print("\nTesting feature creation...")
    sample_data = {
        'invocation_rate': 50.0,
        'memory_mb': 1024.0,
        'duration': 0.5,
        'total_latency_ms': 200.0,
        'hour': 14
    }
    features = create_operational_features(sample_data)
    print(f"  Sample features: {features}")
    print(f"  Feature ranges: min={features.min():.3f}, max={features.max():.3f}")

    print("\nModel test successful!")
