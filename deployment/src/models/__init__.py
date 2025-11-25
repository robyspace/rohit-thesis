"""
Multi-Cloud Serverless Orchestration - Model Definitions
=========================================================

This package contains the PyTorch model architectures for the three-layer
hierarchical deep reinforcement learning framework:

1. Strategic Layer (DQN): Cloud provider selection
2. Tactical Layer (PPO): Regional placement and memory allocation
3. Operational Layer (LSTM): Real-time resource scaling prediction

Models:
-------
- EnhancedDQNetwork: DQN for strategic cloud provider selection
- PPOActorCritic: PPO for tactical function placement
- LSTMPredictor: LSTM for operational workload prediction
- AsymmetricMSELoss: Custom loss function for LSTM training

Usage:
------
>>> from models.dqn_strategic import EnhancedDQNetwork, load_dqn_model
>>> from models.ppo_tactical import PPOActorCritic, load_ppo_model
>>> from models.lstm_operational import LSTMPredictor, load_lstm_model
>>>
>>> # Load models
>>> dqn_model = load_dqn_model('path/to/dqn_checkpoint.pt')
>>> ppo_model = load_ppo_model('path/to/ppo_checkpoint.pt')
>>> lstm_model = load_lstm_model('path/to/lstm_checkpoint.pt')
"""

from .dqn_strategic import EnhancedDQNetwork, create_dqn_model, load_dqn_model
from .ppo_tactical import (
    PPOActorCritic,
    create_ppo_model,
    load_ppo_model,
    decode_action,
    encode_action,
    REGIONS,
    MEMORY_TIERS
)
from .lstm_operational import (
    LSTMPredictor,
    AsymmetricMSELoss,
    create_lstm_model,
    load_lstm_model,
    create_operational_features
)

__all__ = [
    # DQN Strategic
    'EnhancedDQNetwork',
    'create_dqn_model',
    'load_dqn_model',
    # PPO Tactical
    'PPOActorCritic',
    'create_ppo_model',
    'load_ppo_model',
    'decode_action',
    'encode_action',
    'REGIONS',
    'MEMORY_TIERS',
    # LSTM Operational
    'LSTMPredictor',
    'AsymmetricMSELoss',
    'create_lstm_model',
    'load_lstm_model',
    'create_operational_features',
]

__version__ = '1.0.0'
__author__ = 'Rohit'
