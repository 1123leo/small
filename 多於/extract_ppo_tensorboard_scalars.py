#!/usr/bin/env python3
"""Extract metrics from TensorBoard event files"""

import os
from pathlib import Path
import struct
import json
from tensorboard.compat.proto.event_pb2 import Event

def read_event_file(file_path):
    """Read TensorBoard event file and extract scalar data"""
    scalars = {}
    try:
        with open(file_path, 'rb') as f:
            while True:
                # Read length prefix
                length_bytes = f.read(8)
                if len(length_bytes) < 8:
                    break
                length = struct.unpack('<Q', length_bytes)[0]
                
                # Read event data
                data = f.read(length)
                if len(data) < length:
                    break
                
                event = Event()
                event.ParseFromString(data)
                
                # Extract scalar values
                if event.summary.value:
                    for value in event.summary.value:
                        if value.HasField('simple_value'):
                            key = value.tag
                            val = value.simple_value
                            step = event.step
                            if key not in scalars:
                                scalars[key] = []
                            scalars[key].append((step, val))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return scalars

def extract_metrics(tb_dir):
    """Extract key metrics from TB directory"""
    results = {}
    
    if not Path(tb_dir).exists():
        return None
    
    for event_file in sorted(Path(tb_dir).glob('events.out.*'))[:1]:
        scalars = read_event_file(str(event_file))
        
        # Extract key metrics
        key_metrics = [
            'rollout/ep_rew_mean',
            'rollout/ep_len_mean', 
            'train/value_loss',
            'train/policy_gradient_loss',
            'train/entropy_loss',
            'train/approx_kl',
            'train/clip_fraction',
        ]
        
        for metric in key_metrics:
            if metric in scalars and scalars[metric]:
                values = scalars[metric]
                # Get first, middle and last values
                first = values[0][1] if values else None
                last = values[-1][1] if values else None
                mid_idx = len(values) // 2
                mid = values[mid_idx][1] if mid_idx < len(values) else None
                
                results[metric] = {
                    'first': float(first) if first else None,
                    'middle': float(mid) if mid else None,
                    'final': float(last) if last else None,
                    'steps': len(values)
                }
    
    return results

# Test on PPO_27
print("=" * 60)
print("Extracting metrics from PPO_27 (likely high-dim)")
print("=" * 60)
ppo27 = extract_metrics('tb_logs/PPO_27')
if ppo27:
    for key, data in sorted(ppo27.items()):
        print(f"\n{key}:")
        print(f"  Initial: {data['first']:.6f}")
        print(f"  Middle:  {data['middle']:.6f}")
        print(f"  Final:   {data['final']:.6f}")
        print(f"  Steps:   {data['steps']}")
else:
    print("No data found")

# Try sweep_1m (large training)
print("\n" + "=" * 60)
print("Extracting metrics from sweep_1m")
print("=" * 60)
sweep = extract_metrics('tb_logs/sweep_1m')
if sweep:
    for key, data in sorted(sweep.items()):
        print(f"\n{key}:")
        print(f"  Initial: {data['first']:.6f}")
        print(f"  Middle:  {data['middle']:.6f}")
        print(f"  Final:   {data['final']:.6f}")
        print(f"  Steps:   {data['steps']}")
else:
    print("No data found")

print("\n" + "=" * 60)
