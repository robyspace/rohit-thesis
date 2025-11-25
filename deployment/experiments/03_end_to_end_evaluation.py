"""
Experiment 3: End-to-End Framework Evaluation (Objective 5)

This experiment validates the complete hierarchical DRL framework's
performance on the deployed AWS ECS service.

Metrics:
- Average decision latency (ms)
- Throughput (requests/second)
- Strategic layer inference time (ms)
- Tactical layer inference time (ms)
- Operational layer inference time (ms)
- Memory footprint
- API response time distribution
- Error rate
"""

import sys
import os
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import warnings
warnings.filterwarnings('ignore')


# API endpoint (deployed on AWS ECS)
API_ENDPOINT = os.getenv('API_ENDPOINT', 'http://3.254.111.76:8000')


def load_scaler(scaler_path):
    """Load the RobustScaler used for normalization"""
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)


def denormalize_state(normalized_state, scaler):
    """Denormalize a strategic state using the scaler"""
    # Reshape for scaler
    state_reshaped = normalized_state.reshape(1, -1)
    denormalized = scaler.inverse_transform(state_reshaped)
    return denormalized.squeeze()


def load_test_scenarios(data_path, scaler_path, n_scenarios=500):
    """Load test scenarios from dataset and denormalize them"""
    data = np.load(data_path, allow_pickle=True)
    strategic_states = data['strategic_states']

    # Load scaler
    scaler = load_scaler(scaler_path)

    # Sample random scenarios
    indices = np.random.choice(len(strategic_states), size=n_scenarios, replace=False)

    scenarios = []
    for idx in indices:
        state = strategic_states[idx]

        # Denormalize the state
        denorm_state = denormalize_state(state, scaler)

        scenario = {
            "strategic_state": {
                "hour": max(0, min(23, int(denorm_state[0]))),
                "day_of_week": max(0, min(6, int(denorm_state[1]))),
                "is_weekend": max(0, min(1, int(denorm_state[2]))),
                "is_business_hours": max(0, min(1, int(denorm_state[3]))),
                "invocation_rate": max(0, float(denorm_state[4])),
                "is_bursty": max(0, min(1, int(denorm_state[5]))),
                "avg_duration": max(0, float(denorm_state[6])),
                "avg_cost": max(0, float(denorm_state[7])),
                "avg_carbon": max(0, float(denorm_state[8])),
                "memory_mb": max(128, float(denorm_state[9]))
            },
            "tactical_state": {
                "duration": max(0, float(denorm_state[6])),
                "memory_mb": max(128, float(denorm_state[9])),
                "invocation_rate": max(0, float(denorm_state[4])),
                "cold_start_rate": 0.15,
                "avg_duration": max(0, float(denorm_state[6])),
                "std_duration": max(0, float(denorm_state[6] * 0.1)),
                "is_bursty": max(0, min(1, int(denorm_state[5])))
            },
            "app_profile": {
                "cold_start_rate": 0.15,
                "sla_violation_rate": 0.05,
                "avg_invocation_rate": max(0, float(denorm_state[4])),
                "workload_type": "standard"
            }
        }

        scenarios.append(scenario)

    return scenarios


def test_health_check():
    """Test API health endpoint"""
    print("\n[1/6] Testing API health...")

    try:
        response = requests.get(f"{API_ENDPOINT}/health", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ API is healthy")
            print(f"  Status: {data.get('status')}")
            print(f"  Models loaded: {data.get('models_loaded')}")
            print(f"  Device: {data.get('device')}")
            return True
        else:
            print(f"  ✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Health check error: {e}")
        return False


def make_decision_request(scenario, timeout=30):
    """Make a single decision request to the API"""
    try:
        start_time = time.time()

        response = requests.post(
            f"{API_ENDPOINT}/decision",
            json=scenario,
            timeout=timeout
        )

        latency = (time.time() - start_time) * 1000  # ms

        if response.status_code == 200:
            data = response.json()
            return {
                'success': True,
                'latency': latency,
                'cloud_provider': data.get('cloud_provider'),
                'region': data.get('region'),
                'memory_mb': data.get('memory_mb'),
                'response': data
            }
        else:
            return {
                'success': False,
                'latency': latency,
                'error': f"HTTP {response.status_code}",
                'response': None
            }
    except requests.Timeout:
        return {
            'success': False,
            'latency': timeout * 1000,
            'error': 'Timeout',
            'response': None
        }
    except Exception as e:
        return {
            'success': False,
            'latency': 0,
            'error': str(e),
            'response': None
        }


def test_sequential_requests(scenarios, n_requests=50):
    """Test sequential request processing"""
    print(f"\n[2/6] Testing sequential requests ({n_requests} requests)...")

    results = []

    for i, scenario in enumerate(scenarios[:n_requests]):
        if i % 10 == 0:
            print(f"  Processing request {i+1}/{n_requests}...", end='\r')

        result = make_decision_request(scenario)
        results.append(result)

    print()  # New line

    # Calculate metrics
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    if successful:
        latencies = [r['latency'] for r in successful]
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        print(f"  ✓ Success rate: {len(successful)}/{n_requests} ({len(successful)/n_requests*100:.1f}%)")
        print(f"  Avg latency: {avg_latency:.2f} ms")
        print(f"  P50 latency: {p50_latency:.2f} ms")
        print(f"  P95 latency: {p95_latency:.2f} ms")
        print(f"  P99 latency: {p99_latency:.2f} ms")

        if failed:
            print(f"  ✗ Failed requests: {len(failed)}")
            error_types = {}
            for r in failed:
                error = r.get('error', 'Unknown')
                error_types[error] = error_types.get(error, 0) + 1
            for error, count in error_types.items():
                print(f"    - {error}: {count}")
    else:
        print(f"  ✗ All requests failed")
        avg_latency = 0
        p50_latency = 0
        p95_latency = 0
        p99_latency = 0

    return {
        'results': results,
        'avg_latency': avg_latency,
        'p50_latency': p50_latency,
        'p95_latency': p95_latency,
        'p99_latency': p99_latency,
        'success_rate': len(successful) / n_requests if n_requests > 0 else 0,
        'error_rate': len(failed) / n_requests if n_requests > 0 else 1.0
    }


def test_concurrent_requests(scenarios, n_requests=100, max_workers=10):
    """Test concurrent request processing"""
    print(f"\n[3/6] Testing concurrent requests ({n_requests} requests, {max_workers} workers)...")

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(make_decision_request, scenario): i
            for i, scenario in enumerate(scenarios[:n_requests])
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 10 == 0:
                print(f"  Completed {completed}/{n_requests} requests...", end='\r')

            result = future.result()
            results.append(result)

    print()  # New line

    total_time = time.time() - start_time
    throughput = n_requests / total_time

    # Calculate metrics
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    if successful:
        latencies = [r['latency'] for r in successful]
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        print(f"  ✓ Success rate: {len(successful)}/{n_requests} ({len(successful)/n_requests*100:.1f}%)")
        print(f"  Total time: {total_time:.2f} s")
        print(f"  Throughput: {throughput:.2f} req/s")
        print(f"  Avg latency: {avg_latency:.2f} ms")
        print(f"  P95 latency: {p95_latency:.2f} ms")
        print(f"  P99 latency: {p99_latency:.2f} ms")

        if failed:
            print(f"  ✗ Failed requests: {len(failed)}")
    else:
        print(f"  ✗ All requests failed")
        throughput = 0
        avg_latency = 0
        p95_latency = 0
        p99_latency = 0

    return {
        'results': results,
        'throughput': throughput,
        'total_time': total_time,
        'avg_latency': avg_latency,
        'p95_latency': p95_latency,
        'p99_latency': p99_latency,
        'success_rate': len(successful) / n_requests if n_requests > 0 else 0,
        'error_rate': len(failed) / n_requests if n_requests > 0 else 1.0
    }


def test_decision_consistency(scenarios, n_requests=20):
    """Test decision consistency with same input"""
    print(f"\n[4/6] Testing decision consistency ({n_requests} identical requests)...")

    # Use first scenario for all requests
    scenario = scenarios[0]

    results = []
    for i in range(n_requests):
        result = make_decision_request(scenario)
        if result['success']:
            results.append(result)

    if not results:
        print("  ✗ No successful requests")
        return {'consistent': False, 'decisions': []}

    # Check consistency
    decisions = [
        (r['cloud_provider'], r['region'], r['memory_mb'])
        for r in results
    ]

    unique_decisions = set(decisions)

    print(f"  ✓ Successful requests: {len(results)}/{n_requests}")
    print(f"  Unique decisions: {len(unique_decisions)}")

    if len(unique_decisions) == 1:
        print(f"  ✓ Decisions are consistent")
        print(f"    Cloud: {results[0]['cloud_provider']}")
        print(f"    Region: {results[0]['region']}")
        print(f"    Memory: {results[0]['memory_mb']} MB")
    else:
        print(f"  ⚠ Decisions vary:")
        for decision, count in pd.Series(decisions).value_counts().items():
            cloud, region, memory = decision
            print(f"    {cloud}/{region}/{memory}MB: {count} times")

    return {
        'consistent': len(unique_decisions) == 1,
        'decisions': decisions,
        'unique_count': len(unique_decisions)
    }


def test_different_workloads(scenarios):
    """Test framework with different workload patterns"""
    print(f"\n[5/6] Testing different workload patterns...")

    # Categorize scenarios by workload characteristics
    workload_types = {
        'low_traffic': [],
        'high_traffic': [],
        'bursty': [],
        'steady': []
    }

    for scenario in scenarios[:100]:
        invocation_rate = scenario['strategic_state']['invocation_rate']
        is_bursty = scenario['strategic_state']['is_bursty']

        if invocation_rate < 0.3:
            workload_types['low_traffic'].append(scenario)
        if invocation_rate > 0.7:
            workload_types['high_traffic'].append(scenario)
        if is_bursty == 1:
            workload_types['bursty'].append(scenario)
        if is_bursty == 0:
            workload_types['steady'].append(scenario)

    workload_results = {}

    for workload_type, workload_scenarios in workload_types.items():
        if not workload_scenarios:
            continue

        print(f"\n  Testing {workload_type} ({len(workload_scenarios)} scenarios)...")

        results = []
        for scenario in workload_scenarios[:20]:  # Test 20 of each type
            result = make_decision_request(scenario)
            if result['success']:
                results.append(result)

        if results:
            latencies = [r['latency'] for r in results]
            decisions = [(r['cloud_provider'], r['region']) for r in results]

            avg_latency = np.mean(latencies)
            decision_diversity = len(set(decisions))

            print(f"    Avg latency: {avg_latency:.2f} ms")
            print(f"    Decision diversity: {decision_diversity} unique placements")

            workload_results[workload_type] = {
                'avg_latency': avg_latency,
                'decision_diversity': decision_diversity,
                'success_rate': len(results) / len(workload_scenarios[:20])
            }

    return workload_results


def plot_results(sequential_results, concurrent_results, output_dir):
    """Create visualizations"""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Sequential latency distribution
    ax1 = axes[0, 0]
    successful = [r for r in sequential_results['results'] if r['success']]
    if successful:
        latencies = [r['latency'] for r in successful]
        ax1.hist(latencies, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(latencies), color='red', linestyle='--', label=f'Mean: {np.mean(latencies):.2f}ms')
        ax1.axvline(np.percentile(latencies, 95), color='orange', linestyle='--', label=f'P95: {np.percentile(latencies, 95):.2f}ms')
        ax1.axvline(np.percentile(latencies, 99), color='darkred', linestyle='--', label=f'P99: {np.percentile(latencies, 99):.2f}ms')
        ax1.set_xlabel('Latency (ms)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Sequential Request Latency Distribution', fontsize=14, fontweight='bold')
        ax1.legend()

    # Plot 2: Concurrent latency distribution
    ax2 = axes[0, 1]
    successful = [r for r in concurrent_results['results'] if r['success']]
    if successful:
        latencies = [r['latency'] for r in successful]
        ax2.hist(latencies, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(latencies), color='red', linestyle='--', label=f'Mean: {np.mean(latencies):.2f}ms')
        ax2.axvline(np.percentile(latencies, 95), color='orange', linestyle='--', label=f'P95: {np.percentile(latencies, 95):.2f}ms')
        ax2.axvline(np.percentile(latencies, 99), color='darkred', linestyle='--', label=f'P99: {np.percentile(latencies, 99):.2f}ms')
        ax2.set_xlabel('Latency (ms)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Concurrent Request Latency Distribution', fontsize=14, fontweight='bold')
        ax2.legend()

    # Plot 3: Latency comparison
    ax3 = axes[1, 0]
    metrics = ['Avg', 'P95', 'P99']
    sequential = [sequential_results['avg_latency'], sequential_results['p95_latency'], sequential_results['p99_latency']]
    concurrent = [concurrent_results['avg_latency'], concurrent_results['p95_latency'], concurrent_results['p99_latency']]

    x = np.arange(len(metrics))
    width = 0.35

    ax3.bar(x - width/2, sequential, width, label='Sequential', color='#3498db')
    ax3.bar(x + width/2, concurrent, width, label='Concurrent', color='#2ecc71')
    ax3.set_ylabel('Latency (ms)', fontsize=12)
    ax3.set_title('Latency Comparison: Sequential vs Concurrent', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()

    # Add value labels on bars
    for i, (seq, conc) in enumerate(zip(sequential, concurrent)):
        ax3.text(i - width/2, seq, f'{seq:.1f}', ha='center', va='bottom', fontsize=9)
        ax3.text(i + width/2, conc, f'{conc:.1f}', ha='center', va='bottom', fontsize=9)

    # Plot 4: Success rate and throughput
    ax4 = axes[1, 1]
    metrics = ['Success Rate (%)', 'Error Rate (%)']
    sequential_rates = [sequential_results['success_rate'] * 100, sequential_results['error_rate'] * 100]
    concurrent_rates = [concurrent_results['success_rate'] * 100, concurrent_results['error_rate'] * 100]

    x = np.arange(len(metrics))

    ax4.bar(x - width/2, sequential_rates, width, label='Sequential', color='#3498db')
    ax4.bar(x + width/2, concurrent_rates, width, label='Concurrent', color='#2ecc71')
    ax4.set_ylabel('Percentage (%)', fontsize=12)
    ax4.set_title('Reliability Metrics', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.set_ylim([0, 105])

    # Add value labels
    for i, (seq, conc) in enumerate(zip(sequential_rates, concurrent_rates)):
        ax4.text(i - width/2, seq, f'{seq:.1f}%', ha='center', va='bottom', fontsize=9)
        ax4.text(i + width/2, conc, f'{conc:.1f}%', ha='center', va='bottom', fontsize=9)

    # Add throughput annotation
    ax4.text(0.5, 90, f'Throughput: {concurrent_results["throughput"]:.2f} req/s',
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'end_to_end_evaluation.png'), dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_dir}/end_to_end_evaluation.png")


def create_summary_table(sequential_results, concurrent_results, output_dir):
    """Create results summary table"""
    summary_data = {
        'Test Type': ['Sequential', 'Concurrent'],
        'Avg Latency (ms)': [
            f"{sequential_results['avg_latency']:.2f}",
            f"{concurrent_results['avg_latency']:.2f}"
        ],
        'P95 Latency (ms)': [
            f"{sequential_results['p95_latency']:.2f}",
            f"{concurrent_results['p95_latency']:.2f}"
        ],
        'P99 Latency (ms)': [
            f"{sequential_results['p99_latency']:.2f}",
            f"{concurrent_results['p99_latency']:.2f}"
        ],
        'Success Rate (%)': [
            f"{sequential_results['success_rate']*100:.1f}",
            f"{concurrent_results['success_rate']*100:.1f}"
        ],
        'Throughput (req/s)': [
            'N/A',
            f"{concurrent_results['throughput']:.2f}"
        ]
    }

    df = pd.DataFrame(summary_data)

    # Save to CSV
    csv_path = os.path.join(output_dir, 'end_to_end_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results table saved to: {csv_path}")

    # Print table
    print("\n" + "="*100)
    print("EXPERIMENT 3: END-TO-END FRAMEWORK EVALUATION RESULTS")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)

    return df


def main():
    """Run end-to-end evaluation experiment"""
    print("="*100)
    print("EXPERIMENT 3: END-TO-END FRAMEWORK EVALUATION")
    print("="*100)
    print(f"\nAPI Endpoint: {API_ENDPOINT}")

    # Paths
    data_path = '../../datasets/processed/drl_states_actions_CORRECTED.npz'
    scaler_path = '../../datasets/processed/robust_scaler.pkl'
    output_dir = '../results'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Test health
    if not test_health_check():
        print("\n✗ API health check failed. Please ensure the API is running.")
        print("  Try: curl http://3.254.111.76:8000/health")
        return

    # Load test scenarios
    print("\nLoading test scenarios...")
    scenarios = load_test_scenarios(data_path, scaler_path, n_scenarios=500)
    print(f"✓ Loaded {len(scenarios)} test scenarios")

    # Run tests
    sequential_results = test_sequential_requests(scenarios, n_requests=50)
    concurrent_results = test_concurrent_requests(scenarios, n_requests=100, max_workers=10)
    consistency_results = test_decision_consistency(scenarios, n_requests=20)
    workload_results = test_different_workloads(scenarios)

    # Create visualizations
    print("\n[6/6] Creating visualizations...")
    plot_results(sequential_results, concurrent_results, output_dir)

    # Create summary table
    create_summary_table(sequential_results, concurrent_results, output_dir)

    # Print summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"Sequential Performance:")
    print(f"  Avg Latency: {sequential_results['avg_latency']:.2f} ms")
    print(f"  P99 Latency: {sequential_results['p99_latency']:.2f} ms")
    print(f"  Success Rate: {sequential_results['success_rate']*100:.1f}%")
    print(f"\nConcurrent Performance:")
    print(f"  Throughput: {concurrent_results['throughput']:.2f} req/s")
    print(f"  Avg Latency: {concurrent_results['avg_latency']:.2f} ms")
    print(f"  P99 Latency: {concurrent_results['p99_latency']:.2f} ms")
    print(f"  Success Rate: {concurrent_results['success_rate']*100:.1f}%")
    print(f"\nDecision Consistency:")
    print(f"  Consistent: {consistency_results['consistent']}")
    print(f"  Unique decisions: {consistency_results['unique_count']}")
    print("="*100)
    print("✓ Experiment 3 complete!")
    print("="*100)


if __name__ == "__main__":
    main()
