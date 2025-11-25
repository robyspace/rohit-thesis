"""
Experiment 1: Cost Prediction Accuracy (Objective 3)

This experiment evaluates the LSTM operational layer's ability to predict
workload characteristics and associated costs compared to baseline models.

Metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Prediction latency
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import torch
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.lstm_operational import LSTMPredictor


class SimpleMovingAverage:
    """Baseline 1: Simple Moving Average predictor"""
    def __init__(self, window_size=3, output_dim=3):
        self.window_size = window_size
        self.output_dim = output_dim

    def predict(self, sequence):
        """Predict using average of last window_size timesteps"""
        if len(sequence) < self.window_size:
            avg = np.mean(sequence, axis=0)
        else:
            avg = np.mean(sequence[-self.window_size:], axis=0)
        # Return only first output_dim features
        return avg[:self.output_dim]


class LinearRegressionPredictor:
    """Baseline 2: Linear Regression predictor"""
    def __init__(self):
        self.models = [LinearRegression() for _ in range(3)]
        self.is_fitted = False

    def fit(self, X_sequences, y_targets):
        """Fit linear models for each output dimension"""
        # Flatten sequences: (batch, seq_len, features) -> (batch, seq_len*features)
        X_flat = X_sequences.reshape(X_sequences.shape[0], -1)

        for i, model in enumerate(self.models):
            model.fit(X_flat, y_targets[:, i])

        self.is_fitted = True

    def predict(self, sequence):
        """Predict using fitted linear models"""
        if not self.is_fitted:
            return np.zeros(3)

        # Flatten sequence
        X_flat = sequence.reshape(1, -1)
        predictions = np.array([model.predict(X_flat)[0] for model in self.models])
        return predictions


class ARIMAPredictor:
    """Baseline 3: ARIMA predictor"""
    def __init__(self, order=(1, 0, 1), output_dim=3):
        self.order = order
        self.output_dim = output_dim

    def predict(self, sequence):
        """Predict using ARIMA for each feature"""
        predictions = []

        # Only predict first output_dim features
        for feature_idx in range(min(self.output_dim, sequence.shape[1])):
            time_series = sequence[:, feature_idx]

            try:
                # Fit ARIMA model
                model = ARIMA(time_series, order=self.order)
                fitted = model.fit()

                # Forecast 1 step ahead
                forecast = fitted.forecast(steps=1)[0]
                predictions.append(forecast)
            except:
                # If ARIMA fails, use last value
                predictions.append(time_series[-1])

        return np.array(predictions)


def load_lstm_model(model_path, device):
    """Load trained LSTM model"""
    model = LSTMPredictor(
        input_dim=5,
        hidden_dim1=128,
        hidden_dim2=64,
        output_dim=3,
        dropout=0.2
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_test_data(data_path):
    """Load test sequences from processed dataset"""
    data = np.load(data_path, allow_pickle=True)

    # Extract operational sequences and targets
    # Assuming the NPZ file contains sequences for LSTM training
    if 'operational_sequences' in data:
        sequences = data['operational_sequences']
        targets = data['operational_targets']
    else:
        # If not available, create from strategic/tactical states
        print("Warning: No pre-split test data found. Creating test sequences...")

        # Load full dataset
        strategic_states = data['strategic_states']

        # Use last 10000 samples as test set to avoid index error
        test_size = min(10000, int(0.1 * len(strategic_states)))

        # For operational layer, we need sequences
        # Extract features from strategic states
        # Strategic state format (10 dims): hour, day_of_week, is_weekend, is_business_hours,
        #                                   invocation_rate, is_bursty, avg_duration, avg_cost, avg_carbon, memory_mb
        # We need 5 features for operational LSTM: invocation_rate(4), duration(6), memory(9), cost(7), carbon(8)
        feature_indices = [4, 6, 9, 7, 8]  # invocation_rate, avg_duration, memory_mb, avg_cost, avg_carbon

        sequences = []
        targets = []

        # Create sequences of length 12
        seq_length = 12
        start_idx = len(strategic_states) - test_size - seq_length
        end_idx = len(strategic_states) - seq_length

        for i in range(start_idx, end_idx):
            seq = strategic_states[i:i+seq_length][:, feature_indices]
            target = strategic_states[i+seq_length][feature_indices[:3]]  # Predict next: invocation_rate, duration, memory
            sequences.append(seq)
            targets.append(target)

        sequences = np.array(sequences)
        targets = np.array(targets)

        print(f"  Created {len(sequences)} test sequences from dataset")

    return sequences, targets


def evaluate_model(model, sequences, targets, device, model_name="Model"):
    """Evaluate a model on test sequences"""
    predictions = []
    inference_times = []

    print(f"\nEvaluating {model_name}...")

    if isinstance(model, LSTMPredictor):
        # PyTorch LSTM model
        with torch.no_grad():
            for seq in sequences:
                start_time = time.time()

                # Add batch dimension
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
                pred = model(seq_tensor).cpu().numpy().squeeze()

                inference_time = (time.time() - start_time) * 1000  # ms

                predictions.append(pred)
                inference_times.append(inference_time)

    elif isinstance(model, LinearRegressionPredictor):
        # Linear regression (requires fitting first)
        model.fit(sequences, targets)

        for seq in sequences:
            start_time = time.time()
            pred = model.predict(seq)
            inference_time = (time.time() - start_time) * 1000  # ms

            predictions.append(pred)
            inference_times.append(inference_time)

    else:
        # Baseline models (SMA, ARIMA)
        for seq in sequences:
            start_time = time.time()
            pred = model.predict(seq)
            inference_time = (time.time() - start_time) * 1000  # ms

            predictions.append(pred)
            inference_times.append(inference_time)

    predictions = np.array(predictions)

    # Calculate metrics
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    avg_latency = np.mean(inference_times)
    p99_latency = np.percentile(inference_times, 99)

    # Per-feature metrics
    feature_names = ['Request Rate', 'Memory Util', 'CPU Util']
    per_feature_mae = [mean_absolute_error(targets[:, i], predictions[:, i])
                       for i in range(3)]

    results = {
        'model': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'avg_latency_ms': avg_latency,
        'p99_latency_ms': p99_latency,
        'per_feature_mae': dict(zip(feature_names, per_feature_mae)),
        'predictions': predictions
    }

    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Avg Latency: {avg_latency:.2f} ms")
    print(f"  P99 Latency: {p99_latency:.2f} ms")

    return results


def plot_results(all_results, output_dir):
    """Create comparison visualizations"""

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)

    # Extract data
    models = [r['model'] for r in all_results]
    maes = [r['mae'] for r in all_results]
    rmses = [r['rmse'] for r in all_results]
    r2s = [r['r2'] for r in all_results]
    latencies = [r['avg_latency_ms'] for r in all_results]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: MAE Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, maes, color=['#2ecc71' if 'LSTM' in m else '#95a5a6' for m in models])
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax1.set_title('Cost Prediction Accuracy - MAE', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)

    # Plot 2: RMSE Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, rmses, color=['#3498db' if 'LSTM' in m else '#95a5a6' for m in models])
    ax2.set_ylabel('Root Mean Squared Error (RMSE)', fontsize=12)
    ax2.set_title('Cost Prediction Accuracy - RMSE', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)

    # Plot 3: R² Score Comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, r2s, color=['#e74c3c' if 'LSTM' in m else '#95a5a6' for m in models])
    ax3.set_ylabel('R² Score', fontsize=12)
    ax3.set_title('Model Fit Quality - R²', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

    # Plot 4: Inference Latency Comparison
    ax4 = axes[1, 1]
    bars4 = ax4.bar(models, latencies, color=['#9b59b6' if 'LSTM' in m else '#95a5a6' for m in models])
    ax4.set_ylabel('Average Latency (ms)', fontsize=12)
    ax4.set_title('Inference Speed', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)

    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_prediction_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_dir}/cost_prediction_comparison.png")


def create_summary_table(all_results, output_dir):
    """Create results summary table"""

    summary_data = []
    for r in all_results:
        summary_data.append({
            'Model': r['model'],
            'MAE': f"{r['mae']:.4f}",
            'RMSE': f"{r['rmse']:.4f}",
            'R² Score': f"{r['r2']:.3f}",
            'Avg Latency (ms)': f"{r['avg_latency_ms']:.2f}",
            'P99 Latency (ms)': f"{r['p99_latency_ms']:.2f}"
        })

    df = pd.DataFrame(summary_data)

    # Save to CSV
    csv_path = os.path.join(output_dir, 'cost_prediction_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults table saved to: {csv_path}")

    # Print table
    print("\n" + "="*80)
    print("EXPERIMENT 1: COST PREDICTION ACCURACY RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    return df


def main():
    """Run cost prediction accuracy experiment"""

    print("="*80)
    print("EXPERIMENT 1: COST PREDICTION ACCURACY")
    print("="*80)

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Paths
    model_path = '../data/best_lstm_predictor.pt'
    data_path = '../../datasets/processed/drl_states_actions_CORRECTED.npz'
    output_dir = '../results'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load test data
    print("\n[1/5] Loading test data...")
    sequences, targets = load_test_data(data_path)
    print(f"  Test sequences: {sequences.shape}")
    print(f"  Test targets: {targets.shape}")

    # Load LSTM model
    print("\n[2/5] Loading LSTM model...")
    lstm_model = load_lstm_model(model_path, device)
    print("  ✓ LSTM model loaded")

    # Initialize baseline models
    print("\n[3/5] Initializing baseline models...")
    sma_model = SimpleMovingAverage(window_size=3)
    lr_model = LinearRegressionPredictor()
    arima_model = ARIMAPredictor(order=(1, 0, 1))
    print("  ✓ Baselines initialized")

    # Evaluate all models
    print("\n[4/5] Evaluating models...")

    all_results = []

    # LSTM (our model)
    lstm_results = evaluate_model(lstm_model, sequences, targets, device, "LSTM (Ours)")
    all_results.append(lstm_results)

    # Baselines
    sma_results = evaluate_model(sma_model, sequences, targets, device, "Simple Moving Average")
    all_results.append(sma_results)

    lr_results = evaluate_model(lr_model, sequences, targets, device, "Linear Regression")
    all_results.append(lr_results)

    arima_results = evaluate_model(arima_model, sequences[:100], targets[:100], device, "ARIMA (sample)")
    all_results.append(arima_results)

    # Create visualizations
    print("\n[5/5] Creating visualizations...")
    plot_results(all_results, output_dir)

    # Create summary table
    create_summary_table(all_results, output_dir)

    # Calculate improvements
    print("\n" + "="*80)
    print("LSTM IMPROVEMENTS OVER BASELINES")
    print("="*80)

    lstm_mae = lstm_results['mae']
    for r in all_results[1:]:  # Skip LSTM itself
        improvement = ((r['mae'] - lstm_mae) / r['mae']) * 100
        print(f"  vs {r['model']}: {improvement:+.1f}% MAE reduction")

    print("\n" + "="*80)
    print("✓ Experiment 1 complete!")
    print("="*80)


if __name__ == "__main__":
    main()
