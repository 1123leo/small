#!/usr/bin/env python3
"""Extract and compare training metrics from TensorBoard logs"""

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import json

def get_metrics(tb_dir):
    """Extract key metrics from a TensorBoard directory"""
    try:
        ea = EventAccumulator(str(tb_dir), size_guidance={})
        ea.Reload()
        
        metrics = {}
        
        # Key metrics to extract
        key_metrics = [
            'rollout/ep_rew_mean',
            'rollout/ep_len_mean',
            'train/value_loss',
            'train/policy_gradient_loss',
            'train/entropy_loss',
            'train/approx_kl',
            'train/clip_fraction',
            'eval/mean_reward',
            'eval/mean_ep_length',
        ]
        
        for tag in key_metrics:
            try:
                events = ea.Scalars(tag)
                if events:
                    # Extract first, middle, last values
                    steps = [e.step for e in events]
                    values = [e.value for e in events]
                    
                    metrics[tag] = {
                        'first': float(values[0]),
                        'middle': float(values[len(values)//2]) if len(values) > 1 else float(values[0]),
                        'final': float(values[-1]),
                        'steps': len(values),
                        'max': float(max(values)),
                        'min': float(min(values)),
                    }
            except:
                pass
        
        return metrics
    except Exception as e:
        print(f"Error reading {tb_dir}: {e}")
        return None

# Analyze multiple training runs
runs = [
    ('tb_logs/PPO_27', 'PPO_27 (1.7MB)'),
    ('tb_logs/sweep_1m', 'sweep_1m (0.5MB)'),
    ('tb_logs/PPO_1', 'PPO_1 (0.3MB)'),
    ('tb_logs/PPO_249', 'PPO_249 (0.4MB)'),
    ('tb_logs/PPO_250', 'PPO_250 (0.4MB)'),
]

print("=" * 80)
print("FLAPPY BIRD PPO TRAINING METRICS ANALYSIS")
print("=" * 80)

all_results = {}

for tb_dir, label in runs:
    if Path(tb_dir).exists():
        print(f"\n{label}:")
        print("-" * 60)
        metrics = get_metrics(tb_dir)
        
        if metrics:
            all_results[label] = metrics
            
            # Show key metrics
            for key in ['rollout/ep_rew_mean', 'rollout/ep_len_mean', 'train/approx_kl']:
                if key in metrics:
                    m = metrics[key]
                    print(f"  {key:30} | Final: {m['final']:8.4f} | Max: {m['max']:8.4f} | Steps: {m['steps']}")
        else:
            print("  No metrics found")
    else:
        print(f"\n{label}: NOT FOUND")

# Summary comparison
print("\n" + "=" * 80)
print("SUMMARY TABLE: Final Values Comparison")
print("=" * 80)

metrics_to_show = [
    'rollout/ep_rew_mean',
    'rollout/ep_len_mean',
    'train/value_loss',
    'train/approx_kl',
    'train/clip_fraction',
    'train/entropy_loss',
]

print(f"\n{'Metric':<35} | {'PPO_27':>10} | {'sweep_1m':>10} | {'PPO_1':>10}")
print("-" * 75)

for metric in metrics_to_show:
    row = f"{metric:<35} | "
    for label in ['PPO_27 (1.7MB)', 'sweep_1m (0.5MB)', 'PPO_1 (0.3MB)']:
        if label in all_results and metric in all_results[label]:
            val = all_results[label][metric]['final']
            row += f"{val:10.4f} | "
        else:
            row += f"{'N/A':>10} | "
    print(row)

print("\n" + "=" * 80)
print("Analysis complete. Data ready for report insertion.")
print("=" * 80)
