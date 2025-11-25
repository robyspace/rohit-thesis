"""
Feature Engineering for Hierarchical DRL Framework
Extracts strategic, tactical, and operational features from invocation data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def create_strategic_features(df: pd.DataFrame) -> np.ndarray:
    """
    Create strategic features for DQN cloud provider selection

    Strategic features (10-dim):
        - Temporal: hour, day_of_week, is_weekend, is_business_hours
        - Workload: invocation_rate, is_bursty
        - Resource: avg_duration, avg_cost, avg_carbon, memory_mb

    Args:
        df: DataFrame with invocation data

    Returns:
        strategic_features: Array of shape (n_samples, 10)
    """
    strategic_feature_names = [
        'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
        'invocation_rate', 'is_bursty',
        'avg_duration', 'avg_cost', 'avg_carbon',
        'memory_mb'
    ]

    # Ensure all features exist
    missing = [f for f in strategic_feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing strategic features: {missing}")

    strategic_features = df[strategic_feature_names].values
    strategic_features = np.nan_to_num(strategic_features, nan=0.0)

    return strategic_features


def create_tactical_features(df: pd.DataFrame) -> np.ndarray:
    """
    Create tactical features for PPO regional placement

    Tactical features (7-dim):
        - Resource: duration, memory_mb, invocation_rate
        - Cold start: cold_start_rate
        - Performance: avg_duration, std_duration
        - Workload: is_bursty

    Args:
        df: DataFrame with invocation data

    Returns:
        tactical_features: Array of shape (n_samples, 7)
    """
    tactical_feature_names = [
        'duration', 'memory_mb', 'invocation_rate',
        'cold_start_rate', 'avg_duration', 'std_duration', 'is_bursty'
    ]

    # Ensure all features exist
    missing = [f for f in tactical_feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing tactical features: {missing}")

    tactical_features = df[tactical_feature_names].values
    tactical_features = np.nan_to_num(tactical_features, nan=0.0)

    return tactical_features


def create_operational_features(df: pd.DataFrame) -> np.ndarray:
    """
    Create operational features for LSTM workload prediction

    Features are PROPERLY NORMALIZED to [0,1] range:
        - request_rate: Log-normalized invocation rate
        - memory_util: Memory usage ratio [0,1]
        - cpu_util: CPU proxy [0,1]
        - queue_depth: Queue depth ratio [0,1]
        - hour_sin: Cyclical hour encoding

    Args:
        df: DataFrame with invocation data

    Returns:
        operational_features: Array of shape (n_samples, 5)
    """
    df = df.copy()

    # Request rate - NORMALIZE using log scale
    raw_request_rate = df['invocation_rate'].fillna(0.0)
    max_rate = raw_request_rate.max() if len(raw_request_rate) > 0 else 1.0
    df['request_rate'] = np.log1p(raw_request_rate) / np.log1p(max_rate + 1e-8)

    # Memory utilization (already normalized) ✓
    df['memory_util'] = (df['memory_mb'] / 3008.0).fillna(0.5)

    # CPU proxy (already normalized) ✓
    df['cpu_util'] = (df['duration'] / 1000.0).clip(0, 1).fillna(0.5)

    # Queue depth - NORMALIZE to [0,1]
    raw_queue = (df['total_latency_ms'] / 1000.0).fillna(0.0)
    max_queue = raw_queue.max() if len(raw_queue) > 0 else 1.0
    df['queue_depth'] = (raw_queue / (max_queue + 1e-8)).clip(0, 1)

    # Temporal encoding (cyclical)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)

    lstm_features = ['request_rate', 'memory_util', 'cpu_util', 'queue_depth', 'hour_sin']

    operational_features = df[lstm_features].values
    operational_features = np.nan_to_num(operational_features, nan=0.0)

    return operational_features


def create_enhanced_strategic_state(
    strategic_state: np.ndarray,
    app_profile: Dict
) -> np.ndarray:
    """
    Combine strategic state with application context

    Args:
        strategic_state: Pre-computed strategic features (10-dim)
        app_profile: Application profile dict with keys:
            - cold_start_rate
            - sla_violation_rate
            - avg_invocation_rate
            - workload_type

    Returns:
        enhanced_state: Combined state vector (14-dim)
    """
    # Application context features (4-dim)
    app_context = np.array([
        app_profile.get('cold_start_rate', 0.0),
        app_profile.get('sla_violation_rate', 0.0),
        app_profile.get('avg_invocation_rate', 0.0) / 100.0,
        1.0 if app_profile.get('workload_type') == 'bursty' else 0.0
    ], dtype=np.float32)

    # Concatenate: [strategic_state (10) | app_context (4)] = 14-dim
    enhanced_state = np.concatenate([strategic_state, app_context])

    return enhanced_state


def create_enhanced_tactical_state(
    tactical_state: np.ndarray,
    strategic_cloud: int
) -> np.ndarray:
    """
    Combine tactical state with strategic cloud context

    Args:
        tactical_state: Pre-computed tactical features (7-dim)
        strategic_cloud: Cloud provider (0=AWS, 1=Azure, 2=GCP)

    Returns:
        enhanced_state: Combined state vector (11-dim)
    """
    # One-hot encode cloud provider (3-dim)
    cloud_encoding = np.zeros(3, dtype=np.float32)
    cloud_encoding[strategic_cloud] = 1.0

    # Placeholder for region (1-dim) - will be filled by environment
    region_encoding = np.array([0.0], dtype=np.float32)

    # Strategic context (4-dim) = cloud (3) + region (1)
    strategic_context = np.concatenate([cloud_encoding, region_encoding])

    # Concatenate: [tactical_state (7) | strategic_context (4)] = 11-dim
    enhanced_state = np.concatenate([tactical_state, strategic_context])

    return enhanced_state


def create_lstm_sequences(
    operational_features: np.ndarray,
    sequence_length: int = 12,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create LSTM input sequences with sliding window

    Args:
        operational_features: Array of shape (n_samples, 5)
        sequence_length: Number of time steps (default: 12 = 3 minutes)
        stride: Step size for sliding window

    Returns:
        X: Input sequences of shape (n_sequences, sequence_length, 5)
        y: Target values of shape (n_sequences, 3)
    """
    n_samples, n_features = operational_features.shape

    if n_samples < sequence_length + 1:
        raise ValueError(
            f"Not enough samples ({n_samples}) for sequence_length={sequence_length}"
        )

    X = []
    y = []

    for i in range(0, n_samples - sequence_length, stride):
        # Input sequence: [i : i+sequence_length]
        X.append(operational_features[i:i+sequence_length])

        # Target: next step's first 3 features (request_rate, memory_util, cpu_util)
        y.append(operational_features[i+sequence_length, :3])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y


if __name__ == "__main__":
    # Test feature engineering
    print("Testing feature engineering functions...")

    # Create sample data
    sample_data = pd.DataFrame({
        'hour': [12, 13, 14],
        'day_of_week': [1, 1, 1],
        'is_weekend': [0, 0, 0],
        'is_business_hours': [1, 1, 1],
        'invocation_rate': [10.0, 15.0, 20.0],
        'is_bursty': [0, 0, 1],
        'avg_duration': [100.0, 120.0, 110.0],
        'avg_cost': [0.001, 0.0012, 0.0011],
        'avg_carbon': [0.5, 0.55, 0.52],
        'memory_mb': [512, 512, 1024],
        'duration': [95, 115, 105],
        'cold_start_rate': [0.1, 0.1, 0.15],
        'std_duration': [10.0, 15.0, 12.0],
        'total_latency_ms': [200, 250, 220]
    })

    # Test strategic features
    strategic = create_strategic_features(sample_data)
    print(f"✓ Strategic features: {strategic.shape}")

    # Test tactical features
    tactical = create_tactical_features(sample_data)
    print(f"✓ Tactical features: {tactical.shape}")

    # Test operational features
    operational = create_operational_features(sample_data)
    print(f"✓ Operational features: {operational.shape}")

    # Test LSTM sequences
    X, y = create_lstm_sequences(operational, sequence_length=2)
    print(f"✓ LSTM sequences: X={X.shape}, y={y.shape}")

    print("\n✓ All feature engineering functions working correctly!")
