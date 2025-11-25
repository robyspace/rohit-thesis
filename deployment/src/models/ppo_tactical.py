"""
PPO Tactical Layer Model
Proximal Policy Optimization Actor-Critic for function placement in multi-cloud environments.

This module contains the PPOActorCritic class that performs tactical-level decisions
for regional placement and memory allocation across 24 actions (4 regions × 6 memory tiers).
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


class PPOActorCritic(nn.Module):
    """
    PPO Actor-Critic with shared feature extraction for tactical function placement.

    This network implements a policy gradient method for selecting optimal placement
    configurations (region + memory tier) based on tactical state features including
    workload patterns, data locality, and cold start metrics.

    Architecture:
        Shared Feature Extractor: 11 -> 128 (tactical state encoder)
        Actor Network: 128 -> 128 -> 64 -> 24 (policy distribution)
        Critic Network: 128 -> 64 -> 1 (value estimate)

    Args:
        state_dim (int): Input state dimension (default: 11)
            - First 7 features: tactical state (duration, memory_mb, invocation_rate,
              cold_start_rate, avg_duration, std_duration, is_bursty)
            - Last 4 features: strategic context (AWS, Azure, GCP one-hot, normalized region)
        action_dim (int): Number of placement actions (default: 24)
            - 4 regions: us-east-1, us-west-2, eu-west-1, ap-southeast-1
            - 6 memory tiers: 128MB, 256MB, 512MB, 1024MB, 2048MB, 3008MB
        hidden_dim (int): Hidden layer dimension for shared encoder (default: 128)

    Example:
        >>> model = PPOActorCritic(state_dim=11, action_dim=24)
        >>> state = torch.randn(32, 11)  # batch of 32 states
        >>> action, log_prob, value = model.act(state)
        >>> print(f"Action: {action.shape}, Log Prob: {log_prob.shape}, Value: {value.shape}")
    """

    def __init__(self, state_dim=11, action_dim=24, hidden_dim=128):
        super(PPOActorCritic, self).__init__()

        # Shared feature extractor
        # Processes both tactical and strategic context features
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )

        # Actor network (policy)
        # Outputs action logits for categorical distribution over 24 placement actions
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # Critic network (value function)
        # Estimates state value for advantage computation
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using orthogonal initialization.
        This helps stabilize policy gradient training by ensuring proper gradient flow.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state):
        """
        Forward pass through both actor and critic networks.

        Args:
            state (torch.Tensor): Input state tensor of shape (batch_size, state_dim)

        Returns:
            tuple: (action_logits, value)
                - action_logits (torch.Tensor): Unnormalized action probabilities (batch_size, action_dim)
                - value (torch.Tensor): State value estimates (batch_size, 1)
        """
        shared_features = self.shared(state)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_logits, value

    def act(self, state, deterministic=False):
        """
        Sample action from policy and compute log probability and value.

        Args:
            state (torch.Tensor): Input state tensor of shape (batch_size, state_dim)
            deterministic (bool): If True, return argmax action instead of sampling

        Returns:
            tuple: (action, log_prob, value)
                - action (torch.Tensor): Sampled actions (batch_size,)
                - log_prob (torch.Tensor): Log probabilities of actions (batch_size,)
                - value (torch.Tensor): State value estimates (batch_size, 1)
        """
        action_logits, value = self.forward(state)

        # Create categorical distribution over actions
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        if deterministic:
            # Select highest probability action
            action = action_probs.argmax(dim=-1)
        else:
            # Sample from distribution
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate(self, state, action):
        """
        Evaluate action under current policy for PPO update.

        Args:
            state (torch.Tensor): Input state tensor (batch_size, state_dim)
            action (torch.Tensor): Actions to evaluate (batch_size,)

        Returns:
            tuple: (log_prob, value, entropy)
                - log_prob (torch.Tensor): Log probabilities of actions (batch_size,)
                - value (torch.Tensor): State value estimates (batch_size, 1)
                - entropy (torch.Tensor): Policy entropy for exploration (batch_size,)
        """
        action_logits, value = self.forward(state)

        # Create categorical distribution
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        # Compute log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, value, entropy


def create_ppo_model(state_dim=11, action_dim=24, hidden_dim=128, device='cpu'):
    """
    Factory function to create and initialize a PPOActorCritic model.

    Args:
        state_dim (int): Input state dimension
        action_dim (int): Number of placement actions
        hidden_dim (int): Hidden layer dimension
        device (str): Device to place the model on ('cpu' or 'cuda')

    Returns:
        PPOActorCritic: Initialized model on the specified device
    """
    model = PPOActorCritic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    model = model.to(device)
    return model


def load_ppo_model(checkpoint_path, state_dim=11, action_dim=24, hidden_dim=128, device='cpu'):
    """
    Load a trained PPOActorCritic model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint (.pt file)
        state_dim (int): Input state dimension
        action_dim (int): Number of actions
        hidden_dim (int): Hidden layer dimension
        device (str): Device to place the model on

    Returns:
        PPOActorCritic: Loaded model in evaluation mode
    """
    model = create_ppo_model(state_dim, action_dim, hidden_dim, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


# Region and memory tier configurations
REGIONS = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']
MEMORY_TIERS = [128, 256, 512, 1024, 2048, 3008]


def decode_action(action_idx):
    """
    Decode action index to (region, memory_tier) configuration.

    Args:
        action_idx (int): Action index (0-23)

    Returns:
        tuple: (region, memory_tier)
            - region (str): Selected region
            - memory_tier (int): Selected memory in MB
    """
    region_idx = action_idx // len(MEMORY_TIERS)
    memory_idx = action_idx % len(MEMORY_TIERS)
    return REGIONS[region_idx], MEMORY_TIERS[memory_idx]


def encode_action(region, memory_tier):
    """
    Encode (region, memory_tier) configuration to action index.

    Args:
        region (str): Region name
        memory_tier (int): Memory tier in MB

    Returns:
        int: Action index (0-23)
    """
    region_idx = REGIONS.index(region)
    memory_idx = MEMORY_TIERS.index(memory_tier)
    return region_idx * len(MEMORY_TIERS) + memory_idx


if __name__ == "__main__":
    # Test the model
    print("Testing PPOActorCritic...")

    # Create model
    model = PPOActorCritic(state_dim=11, action_dim=24, hidden_dim=128)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Actor parameters: {sum(p.numel() for p in model.actor.parameters()):,}")
    print(f"  Critic parameters: {sum(p.numel() for p in model.critic.parameters()):,}")

    # Test forward pass
    batch_size = 32
    test_state = torch.randn(batch_size, 11)

    # Test act method
    action, log_prob, value = model.act(test_state, deterministic=False)
    print(f"\nAct method:")
    print(f"  State shape: {test_state.shape}")
    print(f"  Action shape: {action.shape}")
    print(f"  Log prob shape: {log_prob.shape}")
    print(f"  Value shape: {value.shape}")

    # Test evaluate method
    log_prob_eval, value_eval, entropy = model.evaluate(test_state, action)
    print(f"\nEvaluate method:")
    print(f"  Log prob shape: {log_prob_eval.shape}")
    print(f"  Value shape: {value_eval.shape}")
    print(f"  Entropy shape: {entropy.shape}")

    # Test action decoding
    sample_action = action[0].item()
    region, memory = decode_action(sample_action)
    print(f"\nSample action {sample_action} decoded to: {region}, {memory}MB")

    print("\nModel test successful!")
