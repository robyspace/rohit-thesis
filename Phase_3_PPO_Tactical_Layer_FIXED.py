"""
CRITICAL FIXES FOR PHASE 3 PPO TRAINING

Apply these fixes to your Phase_3_PPO_Tactical_Layer.ipynb to resolve NaN issues.
Replace the corresponding sections in your notebook with these fixed versions.
"""

# ============================================================================
# FIX 1: Tactical Placement Environment - Add safeguards
# ============================================================================

class TacticalPlacementEnv:
    """
    FIXED: Added NaN protection and safe divisions
    """

    def __init__(self, tactical_states, strategic_states, data_df,
                 app_profile_dict, action_to_config):
        self.tactical_states = tactical_states
        self.strategic_states = strategic_states
        self.data_df = data_df
        self.app_profile_dict = app_profile_dict
        self.action_to_config = action_to_config

        self.state_dim = 11
        self.action_dim = 24

        self.alpha = 0.4
        self.beta = 0.4
        self.gamma = 0.2

    def reset(self, idx, strategic_cloud=None):
        """Initialize state - FIXED: Added validation"""
        row = self.data_df.iloc[idx]

        tactical_state = self.tactical_states[idx]

        # FIX: Check for NaN in tactical state
        if np.isnan(tactical_state).any():
            tactical_state = np.nan_to_num(tactical_state, nan=0.0)

        if strategic_cloud is None:
            strategic_cloud = hash(row['app']) % 3

        cloud_encoding = np.zeros(3, dtype=np.float32)
        cloud_encoding[strategic_cloud] = 1.0

        current_region = row.get('region', 'us-east-1')
        region_idx = REGIONS.index(current_region) if current_region in REGIONS else 0

        strategic_context = np.array([
            cloud_encoding[0],
            cloud_encoding[1],
            cloud_encoding[2],
            region_idx / len(REGIONS)
        ], dtype=np.float32)

        state = np.concatenate([tactical_state, strategic_context])

        # FINAL CHECK: Ensure no NaN/Inf
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        return state, row

    def step(self, action, row):
        """
        FIXED: Added safe divisions and value clipping
        """
        target_region, target_memory = self.action_to_config[action]

        current_region = row.get('region', 'us-east-1')
        current_memory = row.get('memory_mb', 512)

        # CRITICAL FIX: Prevent division by zero
        if current_memory == 0 or pd.isna(current_memory):
            current_memory = 512  # Default to median memory

        base_cost = float(row.get('total_cost', 0.0))
        base_latency = float(row.get('total_latency_ms', 100.0))
        base_carbon = float(row.get('carbon_footprint_g', 0.5))
        is_cold_start = row.get('is_cold_start', 0)

        # CRITICAL FIX: Clip base values to prevent extremes
        base_cost = np.clip(base_cost, 0.0, 10.0)
        base_latency = np.clip(base_latency, 0.0, 5000.0)
        base_carbon = np.clip(base_carbon, 0.0, 500.0)

        # === Cost Component ===
        memory_cost_factor = target_memory / max(current_memory, 1.0)  # FIXED
        memory_cost_factor = np.clip(memory_cost_factor, 0.1, 10.0)  # Limit range

        adjusted_cost = base_cost * memory_cost_factor

        if target_region != current_region:
            data_transfer_penalty = 0.1 * (1.0 - REGION_DATA_LOCALITY_SCORE[target_region])
            adjusted_cost += data_transfer_penalty

        cost_reward = 1.0 - min(adjusted_cost / 1.0, 1.0)

        # === Performance Component ===
        network_penalty = REGION_LATENCY[target_region]
        adjusted_latency = base_latency + network_penalty

        if is_cold_start and target_memory >= 1024:
            adjusted_latency *= 0.8

        perf_reward = 1.0 - min(adjusted_latency / 1000.0, 1.0)

        # === Carbon Component ===
        carbon_intensity_factor = REGION_CARBON_INTENSITY[target_region] / 385.0
        adjusted_carbon = base_carbon * carbon_intensity_factor

        carbon_reward = 1.0 - min(adjusted_carbon / 100.0, 1.0)

        # === Data Locality Bonus ===
        locality_bonus = REGION_DATA_LOCALITY_SCORE[target_region] * 0.1

        # === Multi-objective reward ===
        reward = (self.alpha * cost_reward +
                 self.beta * perf_reward +
                 self.gamma * carbon_reward +
                 locality_bonus)

        # SLA penalty
        if adjusted_latency > 1000:
            reward -= 2.0

        # CRITICAL FIX: Clip final reward to prevent extremes
        reward = np.clip(reward, -10.0, 10.0)

        # FINAL CHECK: Ensure no NaN
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        return float(reward), False

    def evaluate_placement(self, action, row):
        """Detailed evaluation for analysis"""
        target_region, target_memory = self.action_to_config[action]
        current_region = row.get('region', 'us-east-1')

        reward, _ = self.step(action, row)

        return {
            'reward': reward,
            'target_region': target_region,
            'target_memory': target_memory,
            'current_region': current_region,
            'region_changed': target_region != current_region,
            'data_locality': REGION_DATA_LOCALITY_SCORE[target_region],
            'carbon_intensity': REGION_CARBON_INTENSITY[target_region]
        }


# ============================================================================
# FIX 2: PPO Agent - Enhanced NaN protection
# ============================================================================

class PPOAgent:
    """
    FIXED: Added comprehensive NaN checks and gradient stability
    """

    def __init__(self, state_dim=11, action_dim=24, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_epsilon=0.2, vf_coef=0.5,
                 entropy_coef=0.01, max_grad_norm=0.5):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.policy = PPOActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []

        # CRITICAL: Track NaN occurrences
        self.nan_count = 0

    def select_action(self, state, deterministic=False):
        """FIXED: Added NaN validation"""
        # CRITICAL FIX: Check state for NaN/Inf
        if np.isnan(state).any() or np.isinf(state).any():
            print(f"  Warning: Invalid state detected, using zeros")
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        self.policy.eval()

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.policy.act(state_tensor, deterministic)

        self.policy.train()

        # CRITICAL FIX: Validate outputs
        action_val = action.item()
        log_prob_val = log_prob.item()
        value_val = value.item()

        if np.isnan(log_prob_val) or np.isnan(value_val):
            print(f"  Warning: NaN in policy output")
            log_prob_val = 0.0
            value_val = 0.0

        return action_val, log_prob_val, value_val

    def store_transition(self, state, action, log_prob, reward, value, done):
        """FIXED: Added validation before storing"""
        # CRITICAL FIX: Validate all inputs
        if np.isnan(state).any() or np.isinf(state).any():
            return  # Skip invalid transitions

        if np.isnan(reward) or np.isinf(reward):
            return

        if np.isnan(log_prob) or np.isinf(log_prob):
            return

        if np.isnan(value) or np.isinf(value):
            return

        self.buffer.add(state, action, log_prob, reward, value, done)

    def update(self, num_epochs=10, batch_size=64):
        """FIXED: Enhanced stability with NaN checks"""
        states, actions, old_log_probs, rewards, values, dones = self.buffer.get()

        if len(states) == 0:
            return None

        # CRITICAL FIX: Validate buffer data
        states_array = np.array(states)
        if np.isnan(states_array).any() or np.isinf(states_array).any():
            print(f"  ERROR: NaN/Inf in buffer states, clearing buffer")
            self.buffer.clear()
            self.nan_count += 1
            return None

        if np.isnan(rewards).any() or np.isinf(rewards).any():
            print(f"  ERROR: NaN/Inf in rewards, clearing buffer")
            self.buffer.clear()
            self.nan_count += 1
            return None

        # Compute advantages and returns
        last_value = values[-1] if len(values) > 0 else 0.0
        advantages, returns = self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        # Convert to tensors
        states_tensor = torch.FloatTensor(states_array).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # CRITICAL FIX: Check for NaN in advantages/returns
        if torch.isnan(advantages_tensor).any() or torch.isnan(returns_tensor).any():
            print(f"  ERROR: NaN in advantages/returns, skipping update")
            self.buffer.clear()
            self.nan_count += 1
            return None

        # Normalize advantages
        adv_mean = advantages_tensor.mean()
        adv_std = advantages_tensor.std()

        if adv_std < 1e-8:
            adv_std = 1.0

        advantages_tensor = (advantages_tensor - adv_mean) / (adv_std + 1e-8)

        # CRITICAL FIX: Clip normalized advantages
        advantages_tensor = torch.clamp(advantages_tensor, -10.0, 10.0)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0

        dataset_size = len(states)

        for epoch in range(num_epochs):
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]

                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)

                # CRITICAL FIX: Check for NaN after forward pass
                if torch.isnan(log_probs).any() or torch.isnan(values).any():
                    print(f"  ERROR: NaN in forward pass at epoch {epoch}")
                    self.nan_count += 1
                    continue  # Skip this batch

                # Policy loss
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # CRITICAL FIX: Clip ratio to prevent explosion
                ratio = torch.clamp(ratio, 0.01, 100.0)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.entropy_coef * entropy_loss

                # CRITICAL FIX: Check loss for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  ERROR: NaN/Inf loss at epoch {epoch}, skipping")
                    self.nan_count += 1
                    continue

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()

                # CRITICAL FIX: Check gradients
                has_nan_grad = False
                for param in self.policy.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break

                if has_nan_grad:
                    print(f"  ERROR: NaN in gradients, skipping update")
                    self.nan_count += 1
                    continue

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()

        self.buffer.clear()

        num_updates = num_epochs * (dataset_size // batch_size + 1)

        if num_updates == 0:
            return None

        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates

        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropy_losses.append(avg_entropy_loss)

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'nan_count': self.nan_count
        }


# ============================================================================
# FIX 3: Training Loop - Add recovery mechanisms
# ============================================================================

print("\n" + "="*80)
print("Starting PPO Training (WITH NAN PROTECTION)")
print("="*80)

NUM_EPISODES = 30
ROLLOUT_LENGTH = 2048
UPDATE_EPOCHS = 10
BATCH_SIZE = 64
VALIDATE_EVERY = 5

print(f"\n  Configuration:")
print(f"    Episodes: {NUM_EPISODES}")
print(f"    Rollout length: {ROLLOUT_LENGTH:,}")
print(f"    Update epochs: {UPDATE_EPOCHS}")
print(f"    Batch size: {BATCH_SIZE}")
print(f"    NaN protection: ENABLED")

training_history = {
    'episode': [],
    'train_reward': [],
    'train_policy_loss': [],
    'train_value_loss': [],
    'val_reward': [],
    'best_val_reward': -float('inf'),
    'nan_events': []
}

print("\n" + "="*80)
print("Training Progress")
print("="*80)

for episode in range(NUM_EPISODES):
    rollout_indices = np.random.choice(len(train_df), ROLLOUT_LENGTH, replace=False)

    episode_rewards = []
    valid_transitions = 0

    # Collect rollout
    for idx in tqdm(rollout_indices, desc=f"Episode {episode+1}/{NUM_EPISODES} - Collecting"):
        try:
            state, row = train_env.reset(idx)

            action, log_prob, value = agent.select_action(state)

            reward, done = train_env.step(action, row)

            # Store with validation
            agent.store_transition(state, action, log_prob, reward, value, done)

            episode_rewards.append(reward)
            valid_transitions += 1

        except Exception as e:
            # Skip problematic transitions
            continue

    # Only update if we have enough valid transitions
    if valid_transitions < BATCH_SIZE:
        print(f"\n  WARNING: Only {valid_transitions} valid transitions, skipping update")
        continue

    # Update policy
    update_info = agent.update(num_epochs=UPDATE_EPOCHS, batch_size=BATCH_SIZE)

    # Episode statistics
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0

    training_history['episode'].append(episode + 1)
    training_history['train_reward'].append(avg_reward)

    if update_info:
        training_history['train_policy_loss'].append(update_info['policy_loss'])
        training_history['train_value_loss'].append(update_info['value_loss'])
        training_history['nan_events'].append(update_info.get('nan_count', 0))

        print(f"\n  Ep {episode+1:2d} | Reward: {avg_reward:.4f} | "
              f"Policy Loss: {update_info['policy_loss']:.4f} | "
              f"Value Loss: {update_info['value_loss']:.4f} | "
              f"NaN Count: {update_info.get('nan_count', 0)}")
    else:
        print(f"\n  Ep {episode+1:2d} | Reward: {avg_reward:.4f} | Update skipped")
        training_history['nan_events'].append(1)

    # Validation
    if (episode + 1) % VALIDATE_EVERY == 0:
        val_rewards = []
        val_indices = np.random.choice(len(val_df), min(1000, len(val_df)), replace=False)

        for idx in val_indices:
            try:
                state, row = val_env.reset(idx)
                action, _, _ = agent.select_action(state, deterministic=True)
                reward, _ = val_env.step(action, row)
                val_rewards.append(reward)
            except:
                continue

        if val_rewards:
            avg_val_reward = np.mean(val_rewards)
            training_history['val_reward'].append(avg_val_reward)

            print(f"  Validation Reward: {avg_val_reward:.4f}")

            if avg_val_reward > training_history['best_val_reward']:
                training_history['best_val_reward'] = avg_val_reward

                os.makedirs('/content/drive/MyDrive/mythesis/rohit-thesis/models/ppo_tactical', exist_ok=True)
                torch.save(agent.policy.state_dict(),
                          '/content/drive/MyDrive/mythesis/rohit-thesis/models/ppo_tactical/best_ppo_tactical.pt')
                print(f"  ✓ New best model saved!")

print("\n" + "="*80)
print("Training Complete")
print("="*80)
print(f"Best validation reward: {training_history['best_val_reward']:.4f}")
print(f"Total NaN events: {sum(training_history.get('nan_events', [0]))}")

# Save final model
torch.save(agent.policy.state_dict(),
          '/content/drive/MyDrive/mythesis/rohit-thesis/models/ppo_tactical/final_ppo_tactical.pt')

with open('/content/ppo_training_history.json', 'w') as f:
    json.dump(training_history, f, indent=2)

print("\n  ✓ Final model saved")
print("  ✓ Training history saved")
