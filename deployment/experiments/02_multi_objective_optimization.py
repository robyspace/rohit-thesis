"""
Experiment 2: Multi-Objective Optimization (Objective 2)

This experiment validates the hierarchical DRL framework's ability to balance
multiple objectives: cost (40%), performance (40%), and carbon footprint (20%).

Metrics:
- Average cost per invocation ($)
- P99 latency (ms)
- Carbon emissions (gCO2e)
- Weighted multi-objective score
- Pareto efficiency
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.hierarchical_coordinator import HierarchicalCoordinator


# Cloud provider cost and carbon data (average per region)
CLOUD_PRICING = {
    'AWS': {
        'us-east-1': {'cost_per_gb_sec': 0.0000166667, 'cost_per_invocation': 0.0000002, 'carbon_intensity': 415},  # gCO2e/kWh
        'us-west-2': {'cost_per_gb_sec': 0.0000166667, 'cost_per_invocation': 0.0000002, 'carbon_intensity': 320},
        'eu-west-1': {'cost_per_gb_sec': 0.0000166667, 'cost_per_invocation': 0.0000002, 'carbon_intensity': 290},
        'ap-southeast-1': {'cost_per_gb_sec': 0.0000166667, 'cost_per_invocation': 0.0000002, 'carbon_intensity': 500}
    },
    'Azure': {
        'us-east-1': {'cost_per_gb_sec': 0.0000160, 'cost_per_invocation': 0.000000167, 'carbon_intensity': 415},
        'us-west-2': {'cost_per_gb_sec': 0.0000160, 'cost_per_invocation': 0.000000167, 'carbon_intensity': 320},
        'eu-west-1': {'cost_per_gb_sec': 0.0000160, 'cost_per_invocation': 0.000000167, 'carbon_intensity': 290},
        'ap-southeast-1': {'cost_per_gb_sec': 0.0000160, 'cost_per_invocation': 0.000000167, 'carbon_intensity': 500}
    },
    'GCP': {
        'us-east-1': {'cost_per_gb_sec': 0.0000025, 'cost_per_invocation': 0.0000004, 'carbon_intensity': 415},
        'us-west-2': {'cost_per_gb_sec': 0.0000025, 'cost_per_invocation': 0.0000004, 'carbon_intensity': 320},
        'eu-west-1': {'cost_per_gb_sec': 0.0000025, 'cost_per_invocation': 0.0000004, 'carbon_intensity': 290},
        'ap-southeast-1': {'cost_per_gb_sec': 0.0000025, 'cost_per_invocation': 0.0000004, 'carbon_intensity': 500}
    }
}

# Expected latency by cloud provider and region (ms) - P99
LATENCY_DATA = {
    'AWS': {'us-east-1': 85, 'us-west-2': 90, 'eu-west-1': 100, 'ap-southeast-1': 120},
    'Azure': {'us-east-1': 90, 'us-west-2': 95, 'eu-west-1': 105, 'ap-southeast-1': 125},
    'GCP': {'us-east-1': 80, 'us-west-2': 85, 'eu-west-1': 95, 'ap-southeast-1': 115}
}


def calculate_cost(cloud, region, memory_mb, duration_ms):
    """Calculate cost for a single invocation"""
    pricing = CLOUD_PRICING[cloud][region]
    memory_gb = memory_mb / 1024
    duration_sec = duration_ms / 1000

    compute_cost = memory_gb * duration_sec * pricing['cost_per_gb_sec']
    invocation_cost = pricing['cost_per_invocation']

    return compute_cost + invocation_cost


def calculate_carbon(cloud, region, memory_mb, duration_ms):
    """Calculate carbon footprint (gCO2e) for a single invocation"""
    carbon_intensity = CLOUD_PRICING[cloud][region]['carbon_intensity']

    # Estimate power consumption: ~2W per GB memory
    memory_gb = memory_mb / 1024
    power_watts = 2.0 * memory_gb
    duration_hours = duration_ms / (1000 * 3600)

    # Energy (kWh) = Power (W) * Time (h) / 1000
    energy_kwh = power_watts * duration_hours / 1000

    # Carbon (gCO2e) = Energy (kWh) * Carbon Intensity (gCO2e/kWh)
    carbon_g = energy_kwh * carbon_intensity

    return carbon_g


def get_latency(cloud, region, memory_mb):
    """Get expected P99 latency"""
    base_latency = LATENCY_DATA[cloud][region]

    # Memory affects cold start latency
    memory_factor = 1.0 + (1.0 - memory_mb / 1024)  # Lower memory = higher latency

    return base_latency * memory_factor


class BaselineOptimizer:
    """Base class for baseline strategies"""
    def __init__(self, name):
        self.name = name

    def make_decision(self, scenario):
        raise NotImplementedError


class CostOnlyOptimizer(BaselineOptimizer):
    """Baseline 1: Minimize cost only"""
    def __init__(self):
        super().__init__("Cost-Only")

    def make_decision(self, scenario):
        # Always choose GCP (cheapest), us-east-1, minimum memory
        return {
            'cloud_provider': 'GCP',
            'region': 'us-east-1',
            'memory_mb': 128
        }


class PerformanceOnlyOptimizer(BaselineOptimizer):
    """Baseline 2: Minimize latency only"""
    def __init__(self):
        super().__init__("Performance-Only")

    def make_decision(self, scenario):
        # Choose GCP (lowest latency), eu-west-1, max memory for best perf
        return {
            'cloud_provider': 'GCP',
            'region': 'us-east-1',
            'memory_mb': 1024
        }


class CarbonOnlyOptimizer(BaselineOptimizer):
    """Baseline 3: Minimize carbon only"""
    def __init__(self):
        super().__init__("Carbon-Only")

    def make_decision(self, scenario):
        # Choose eu-west-1 (lowest carbon), minimum memory
        return {
            'cloud_provider': 'AWS',
            'region': 'eu-west-1',
            'memory_mb': 128
        }


class RandomOptimizer(BaselineOptimizer):
    """Baseline 4: Random selection"""
    def __init__(self):
        super().__init__("Random")
        self.clouds = ['AWS', 'Azure', 'GCP']
        self.regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']
        self.memory_options = [128, 256, 512, 1024, 2048, 3008]

    def make_decision(self, scenario):
        return {
            'cloud_provider': np.random.choice(self.clouds),
            'region': np.random.choice(self.regions),
            'memory_mb': np.random.choice(self.memory_options)
        }


def load_test_scenarios(data_path, n_scenarios=1000):
    """Load test scenarios from dataset"""
    data = np.load(data_path, allow_pickle=True)
    strategic_states = data['strategic_states']

    # Sample random scenarios
    indices = np.random.choice(len(strategic_states), size=n_scenarios, replace=False)

    scenarios = []
    for idx in indices:
        state = strategic_states[idx]

        scenario = {
            'strategic_state': state,
            'tactical_state': np.array([
                state[6],  # duration
                state[9],  # memory_mb
                state[4],  # invocation_rate
                0.15,  # cold_start_rate (estimated)
                state[6],  # avg_duration
                state[6] * 0.1,  # std_duration (estimated)
                state[5]  # is_bursty
            ]),
            'app_profile': {
                'cold_start_rate': 0.15,
                'sla_violation_rate': 0.05,
                'avg_invocation_rate': state[4],
                'workload_type': 'standard'
            },
            'duration_ms': state[6] if state[6] > 0 else 150.0  # avg_duration
        }

        scenarios.append(scenario)

    return scenarios


def evaluate_strategy(strategy, scenarios, strategy_name):
    """Evaluate a placement strategy on test scenarios"""
    print(f"\nEvaluating {strategy_name}...")

    costs = []
    latencies = []
    carbon_emissions = []
    decision_times = []

    for i, scenario in enumerate(scenarios):
        if i % 100 == 0:
            print(f"  Processing scenario {i}/{len(scenarios)}...", end='\r')

        start_time = time.time()

        # Make decision
        if isinstance(strategy, HierarchicalCoordinator):
            decision = strategy.make_decision(
                strategic_state=scenario['strategic_state'],
                tactical_state=scenario['tactical_state'],
                operational_sequence=None,
                app_profile=scenario['app_profile']
            )
        else:
            decision = strategy.make_decision(scenario)

        decision_time = (time.time() - start_time) * 1000  # ms
        decision_times.append(decision_time)

        # Calculate metrics
        cloud = decision['cloud_provider']
        region = decision['region']
        memory_mb = decision['memory_mb']
        duration_ms = scenario['duration_ms']

        cost = calculate_cost(cloud, region, memory_mb, duration_ms)
        latency = get_latency(cloud, region, memory_mb)
        carbon = calculate_carbon(cloud, region, memory_mb, duration_ms)

        costs.append(cost)
        latencies.append(latency)
        carbon_emissions.append(carbon)

    print()  # New line after progress

    # Calculate aggregate metrics
    avg_cost = np.mean(costs)
    p99_latency = np.percentile(latencies, 99)
    total_carbon = np.sum(carbon_emissions)
    avg_decision_time = np.mean(decision_times)

    # Weighted multi-objective score (lower is better)
    # Normalize each metric to [0, 1] range and weight
    normalized_cost = avg_cost / 0.001  # Normalize to reasonable range
    normalized_latency = p99_latency / 200  # Normalize to reasonable range
    normalized_carbon = total_carbon / 10  # Normalize to reasonable range

    multi_obj_score = (0.4 * normalized_cost +
                       0.4 * normalized_latency +
                       0.2 * normalized_carbon)

    results = {
        'strategy': strategy_name,
        'avg_cost': avg_cost,
        'p99_latency': p99_latency,
        'total_carbon': total_carbon,
        'avg_carbon': np.mean(carbon_emissions),
        'multi_obj_score': multi_obj_score,
        'avg_decision_time': avg_decision_time,
        'costs': costs,
        'latencies': latencies,
        'carbon_emissions': carbon_emissions
    }

    print(f"  Avg Cost: ${avg_cost:.6f}")
    print(f"  P99 Latency: {p99_latency:.2f} ms")
    print(f"  Total Carbon: {total_carbon:.2f} gCO2e")
    print(f"  Multi-Objective Score: {multi_obj_score:.4f}")
    print(f"  Avg Decision Time: {avg_decision_time:.2f} ms")

    return results


def plot_results(all_results, output_dir):
    """Create comparison visualizations"""
    sns.set_style("whitegrid")

    # Extract data
    strategies = [r['strategy'] for r in all_results]
    costs = [r['avg_cost'] * 1000 for r in all_results]  # Convert to cents
    latencies = [r['p99_latency'] for r in all_results]
    carbon = [r['total_carbon'] for r in all_results]
    scores = [r['multi_obj_score'] for r in all_results]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Cost Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(strategies, costs, color=['#2ecc71' if 'DRL' in s else '#95a5a6' for s in strategies])
    ax1.set_ylabel('Average Cost (cents)', fontsize=12)
    ax1.set_title('Cost per Invocation', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}¢',
                ha='center', va='bottom', fontsize=10)

    # Plot 2: Latency Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(strategies, latencies, color=['#3498db' if 'DRL' in s else '#95a5a6' for s in strategies])
    ax2.set_ylabel('P99 Latency (ms)', fontsize=12)
    ax2.set_title('Performance (P99 Latency)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}ms',
                ha='center', va='bottom', fontsize=10)

    # Plot 3: Carbon Footprint
    ax3 = axes[1, 0]
    bars3 = ax3.bar(strategies, carbon, color=['#27ae60' if 'DRL' in s else '#95a5a6' for s in strategies])
    ax3.set_ylabel('Total Carbon (gCO2e)', fontsize=12)
    ax3.set_title('Carbon Footprint', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}g',
                ha='center', va='bottom', fontsize=10)

    # Plot 4: Multi-Objective Score
    ax4 = axes[1, 1]
    bars4 = ax4.bar(strategies, scores, color=['#e74c3c' if 'DRL' in s else '#95a5a6' for s in strategies])
    ax4.set_ylabel('Multi-Objective Score (lower is better)', fontsize=12)
    ax4.set_title('Balanced Multi-Objective Score (40% cost, 40% perf, 20% carbon)', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_objective_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_dir}/multi_objective_comparison.png")


def create_summary_table(all_results, output_dir):
    """Create results summary table"""
    summary_data = []
    for r in all_results:
        summary_data.append({
            'Strategy': r['strategy'],
            'Avg Cost ($)': f"{r['avg_cost']:.6f}",
            'P99 Latency (ms)': f"{r['p99_latency']:.2f}",
            'Total Carbon (gCO2e)': f"{r['total_carbon']:.2f}",
            'Multi-Obj Score': f"{r['multi_obj_score']:.4f}",
            'Decision Time (ms)': f"{r['avg_decision_time']:.2f}"
        })

    df = pd.DataFrame(summary_data)

    # Save to CSV
    csv_path = os.path.join(output_dir, 'multi_objective_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults table saved to: {csv_path}")

    # Print table
    print("\n" + "="*100)
    print("EXPERIMENT 2: MULTI-OBJECTIVE OPTIMIZATION RESULTS")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)

    return df


def main():
    """Run multi-objective optimization experiment"""
    print("="*100)
    print("EXPERIMENT 2: MULTI-OBJECTIVE OPTIMIZATION")
    print("="*100)

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Paths
    strategic_model_path = '../data/best_enhanced_dqn.pt'
    tactical_model_path = '../data/best_ppo_tactical.pt'
    operational_model_path = '../data/best_lstm_predictor.pt'
    data_path = '../../datasets/processed/drl_states_actions_CORRECTED.npz'
    output_dir = '../results'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load test scenarios
    print("\n[1/4] Loading test scenarios...")
    scenarios = load_test_scenarios(data_path, n_scenarios=1000)
    print(f"  ✓ Loaded {len(scenarios)} test scenarios")

    # Initialize strategies
    print("\n[2/4] Initializing strategies...")

    # Our hierarchical DRL framework
    drl_coordinator = HierarchicalCoordinator(
        strategic_model_path=strategic_model_path,
        tactical_model_path=tactical_model_path,
        operational_model_path=operational_model_path,
        device=device
    )

    # Baselines
    baselines = [
        CostOnlyOptimizer(),
        PerformanceOnlyOptimizer(),
        CarbonOnlyOptimizer(),
        RandomOptimizer()
    ]

    print("  ✓ All strategies initialized")

    # Evaluate all strategies
    print("\n[3/4] Evaluating strategies...")

    all_results = []

    # Evaluate DRL
    drl_results = evaluate_strategy(drl_coordinator, scenarios, "Hierarchical DRL (Ours)")
    all_results.append(drl_results)

    # Evaluate baselines
    for baseline in baselines:
        baseline_results = evaluate_strategy(baseline, scenarios, baseline.name)
        all_results.append(baseline_results)

    # Create visualizations
    print("\n[4/4] Creating visualizations...")
    plot_results(all_results, output_dir)

    # Create summary table
    create_summary_table(all_results, output_dir)

    # Calculate improvements
    print("\n" + "="*100)
    print("HIERARCHICAL DRL IMPROVEMENTS OVER BASELINES")
    print("="*100)

    drl_score = drl_results['multi_obj_score']
    for r in all_results[1:]:
        improvement = ((r['multi_obj_score'] - drl_score) / r['multi_obj_score']) * 100
        print(f"  vs {r['strategy']}: {improvement:+.1f}% multi-objective score improvement")

    print("\n" + "="*100)
    print("✓ Experiment 2 complete!")
    print("="*100)


if __name__ == "__main__":
    main()
