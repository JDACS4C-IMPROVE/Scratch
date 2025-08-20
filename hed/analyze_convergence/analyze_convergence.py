#!/usr/bin/env python3
"""
Convergence Analysis Script for ML Training Runs
Usage: python analyze_convergence.py EXP024
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

def parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object"""
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except:
        return None

def analyze_run(run_path):
    """Analyze a single RUN folder for convergence metrics"""
    epochs_path = run_path / "ckpts" / "epochs"
    
    if not epochs_path.exists():
        return None
    
    epoch_data = []
    
    # Collect all epoch folders
    epoch_folders = [d for d in epochs_path.iterdir() if d.is_dir() and d.name.isdigit()]
    epoch_folders.sort(key=lambda x: int(x.name))
    
    for epoch_dir in epoch_folders:
        ckpt_file = epoch_dir / "ckpt-info.json"
        if not ckpt_file.exists():
            continue
            
        try:
            with open(ckpt_file, 'r') as f:
                data = json.load(f)
                
            epoch_info = {
                'epoch': data.get('epoch', 0),
                'val_loss': data.get('best_metric_last', None),
                'timestamp': data.get('timestamp', ''),
                'time_elapsed': data.get('time_elapsed', ''),
                'save_best_metric': data.get('save_best_metric', ''),
                'model_file': data.get('model_file', '')
            }
            
            epoch_data.append(epoch_info)
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not read {ckpt_file}: {e}")
            continue
    
    if not epoch_data:
        return None
        
    return epoch_data

def calculate_time_stats(epoch_data):
    """Calculate timing statistics between epochs"""
    timestamps = []
    for epoch in epoch_data:
        ts = parse_timestamp(epoch['timestamp'])
        if ts:
            timestamps.append(ts)
    
    if len(timestamps) < 2:
        return {'avg_epoch_time': None, 'total_time': None, 'time_per_epoch': []}
    
    time_diffs = []
    for i in range(1, len(timestamps)):
        diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 60.0  # minutes
        time_diffs.append(diff)
    
    total_time = (timestamps[-1] - timestamps[0]).total_seconds() / 3600.0  # hours
    avg_epoch_time = np.mean(time_diffs) if time_diffs else None
    
    return {
        'avg_epoch_time': avg_epoch_time,
        'total_time': total_time, 
        'time_per_epoch': time_diffs
    }

def classify_convergence(epoch_data, min_epochs=5, plateau_window=5, plateau_threshold=0.001):
    """Classify convergence status based on validation loss trends"""
    
    if len(epoch_data) < min_epochs:
        return 'INSUFFICIENT_DATA', 0.0, {}
    
    # Extract valid losses
    val_losses = [ep['val_loss'] for ep in epoch_data if ep['val_loss'] is not None]
    epochs = [ep['epoch'] for ep in epoch_data if ep['val_loss'] is not None]
    
    if len(val_losses) < min_epochs:
        return 'INSUFFICIENT_DATA', 0.0, {}
    
    val_losses = np.array(val_losses)
    epochs = np.array(epochs)
    
    # Calculate metrics
    initial_loss = val_losses[0]
    final_loss = val_losses[-1]
    min_loss = np.min(val_losses)
    
    # Loss improvement
    improvement = (initial_loss - final_loss) / initial_loss if initial_loss != 0 else 0
    
    # Trend analysis (linear fit)
    if len(val_losses) >= 3:
        slope = np.polyfit(epochs, val_losses, 1)[0]
    else:
        slope = 0
        
    # Plateau detection (check last N epochs)
    plateau_epochs = min(plateau_window, len(val_losses))
    recent_losses = val_losses[-plateau_epochs:]
    plateau_std = np.std(recent_losses) if len(recent_losses) > 1 else float('inf')
    is_plateau = plateau_std < plateau_threshold
    
    # Recent trend (last half of training)
    if len(val_losses) >= 4:
        mid_point = len(val_losses) // 2
        recent_slope = np.polyfit(epochs[mid_point:], val_losses[mid_point:], 1)[0]
    else:
        recent_slope = slope
    
    metrics = {
        'initial_loss': float(initial_loss),
        'final_loss': float(final_loss),
        'min_loss': float(min_loss),
        'improvement_pct': float(improvement * 100),
        'overall_slope': float(slope),
        'recent_slope': float(recent_slope),
        'plateau_std': float(plateau_std),
        'is_plateau': is_plateau,
        'num_epochs': len(val_losses)
    }
    
    # Classification logic
    confidence = 0.0
    
    if improvement > 0.1 and slope < -0.001:  # Strong improvement
        if is_plateau:
            status = 'CONVERGED'
            confidence = 0.9
        else:
            status = 'CONVERGING' 
            confidence = 0.7
    elif improvement > 0.05 and slope < 0:  # Moderate improvement
        if is_plateau:
            status = 'CONVERGED'
            confidence = 0.7
        else:
            status = 'CONVERGING'
            confidence = 0.6
    elif improvement > 0 and abs(slope) < 0.001:  # Minimal improvement, flat
        status = 'STALLED'
        confidence = 0.8
    elif slope > 0.001:  # Getting worse
        status = 'DIVERGING'
        confidence = 0.8
    elif improvement < -0.05:  # Much worse than start
        status = 'DIVERGING'
        confidence = 0.9
    else:
        status = 'UNCERTAIN'
        confidence = 0.3
    
    return status, confidence, metrics

def analyze_experiment(exp_name, base_path=None):
    """Analyze all runs in an experiment"""
    
    if base_path is None:
        base_path = "/lus/flare/projects/candle_aesp_CNDA/out/unorun/Output"
    """Analyze all runs in an experiment"""
    
    exp_path = Path(base_path) / exp_name / "run"
    
    if not exp_path.exists():
        print(f"Error: Experiment path {exp_path} does not exist!")
        return
    
    print(f"Analyzing experiment: {exp_name}")
    print(f"Path: {exp_path}")
    print("=" * 80)
    
    # Find all RUN folders
    run_folders = [d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith('RUN')]
    run_folders.sort(key=lambda x: int(x.name[3:]) if x.name[3:].isdigit() else 999999)
    
    print(f"Found {len(run_folders)} RUN folders")
    print()
    
    results = []
    detailed_results = []
    
    for run_dir in run_folders:
        run_name = run_dir.name
        
        # Analyze this run
        epoch_data = analyze_run(run_dir)
        
        if epoch_data is None:
            result = {
                'run': run_name,
                'status': 'NO_DATA',
                'confidence': 0.0,
                'num_epochs': 0,
                'final_loss': None,
                'improvement_pct': None
            }
        else:
            status, confidence, metrics = classify_convergence(epoch_data)
            time_stats = calculate_time_stats(epoch_data)
            
            result = {
                'run': run_name,
                'status': status,
                'confidence': confidence,
                'num_epochs': metrics.get('num_epochs', 0),
                'final_loss': metrics.get('final_loss'),
                'improvement_pct': metrics.get('improvement_pct'),
                'avg_epoch_time_min': time_stats.get('avg_epoch_time'),
                'total_time_hrs': time_stats.get('total_time')
            }
            
            # Store detailed info for summary
            detailed_result = result.copy()
            detailed_result.update(metrics)
            detailed_result.update(time_stats)
            detailed_result['epoch_data'] = epoch_data
            detailed_results.append(detailed_result)
        
        results.append(result)
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Generate reports
    generate_executive_summary(df, exp_name)
    print("\n" + "="*80 + "\n")
    generate_detailed_report(detailed_results, exp_name)

def generate_executive_summary(df, exp_name):
    """Generate high-level summary statistics"""
    print("EXECUTIVE SUMMARY")
    print("="*50)
    
    total_runs = len(df)
    
    # Status distribution
    status_counts = df['status'].value_counts()
    print(f"Total Runs Analyzed: {total_runs}")
    print("\nConvergence Status Distribution:")
    for status, count in status_counts.items():
        pct = (count / total_runs) * 100
        print(f"  {status:12}: {count:4d} ({pct:5.1f}%)")
    
    # Performance metrics (excluding NO_DATA and failed runs)
    valid_df = df[df['final_loss'].notna()]
    
    if len(valid_df) > 0:
        print(f"\nPerformance Metrics (n={len(valid_df)}):")
        print(f"  Final Loss    - Mean: {valid_df['final_loss'].mean():.6f}, Std: {valid_df['final_loss'].std():.6f}")
        print(f"  Final Loss    - Min:  {valid_df['final_loss'].min():.6f}, Max: {valid_df['final_loss'].max():.6f}")
        
        improvement_valid = valid_df[valid_df['improvement_pct'].notna()]
        if len(improvement_valid) > 0:
            print(f"  Improvement % - Mean: {improvement_valid['improvement_pct'].mean():.1f}%, Std: {improvement_valid['improvement_pct'].std():.1f}%")
        
        epochs_valid = valid_df[valid_df['num_epochs'] > 0]
        if len(epochs_valid) > 0:
            print(f"  Epochs        - Mean: {epochs_valid['num_epochs'].mean():.1f}, Std: {epochs_valid['num_epochs'].std():.1f}")
        
        time_valid = valid_df[valid_df['avg_epoch_time_min'].notna()]
        if len(time_valid) > 0:
            print(f"  Avg Epoch Time: {time_valid['avg_epoch_time_min'].mean():.1f} ± {time_valid['avg_epoch_time_min'].std():.1f} minutes")
    
    # Best performing runs
    converged_df = df[df['status'].isin(['CONVERGED', 'CONVERGING'])]
    if len(converged_df) > 0:
        best_runs = converged_df.nsmallest(5, 'final_loss')
        print(f"\nTop 5 Best Performing Runs:")
        for _, run in best_runs.iterrows():
            print(f"  {run['run']}: Loss={run['final_loss']:.6f}, Status={run['status']}, Epochs={run['num_epochs']}")

def generate_detailed_report(detailed_results, exp_name, max_display=20):
    """Generate detailed per-run analysis"""
    print("DETAILED RUN ANALYSIS")
    print("="*50)
    
    # Sort by performance (converged first, then by loss)
    def sort_key(x):
        status_priority = {'CONVERGED': 0, 'CONVERGING': 1, 'STALLED': 2, 'UNCERTAIN': 3, 'DIVERGING': 4, 'NO_DATA': 5}
        return (status_priority.get(x['status'], 6), x.get('final_loss', float('inf')))
    
    sorted_results = sorted(detailed_results, key=sort_key)
    
    print(f"Showing top {min(max_display, len(sorted_results))} runs (sorted by performance):")
    print()
    
    # Header
    print(f"{'Run':<8} {'Status':<10} {'Conf':<5} {'Epochs':<7} {'Final Loss':<12} {'Improv%':<8} {'Time(h)':<8}")
    print("-" * 70)
    
    for i, result in enumerate(sorted_results[:max_display]):
        run_name = result['run']
        status = result['status']
        confidence = result.get('confidence', 0)
        epochs = result.get('num_epochs', 0)
        final_loss = result.get('final_loss')
        improvement = result.get('improvement_pct')
        total_time = result.get('total_time_hrs')
        
        final_loss_str = f"{final_loss:.6f}" if final_loss is not None else "N/A"
        improvement_str = f"{improvement:+5.1f}" if improvement is not None else "N/A"
        time_str = f"{total_time:.1f}" if total_time is not None else "N/A"
        
        print(f"{run_name:<8} {status:<10} {confidence:<5.2f} {epochs:<7} {final_loss_str:<12} {improvement_str:<8} {time_str:<8}")
    
    if len(sorted_results) > max_display:
        print(f"\n... and {len(sorted_results) - max_display} more runs")
    
    # Problem runs that need attention
    problem_runs = [r for r in sorted_results if r['status'] in ['DIVERGING', 'NO_DATA']]
    if problem_runs:
        print(f"\nProblem Runs Requiring Attention ({len(problem_runs)} total):")
        for result in problem_runs[:10]:  # Show first 10 problem runs
            print(f"  {result['run']}: {result['status']}")

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python analyze_convergence.py <EXPERIMENT_NAME> [BASE_PATH]")
        print("Example: python analyze_convergence.py EXP024")
        print("Example: python analyze_convergence.py EXP024 /custom/path/to/output")
        print()
        print("Default base path: /lus/flare/projects/candle_aesp_CNDA/out/unorun/Output")
        sys.exit(1)
    
    exp_name = sys.argv[1]
    base_path = sys.argv[2] if len(sys.argv) == 3 else None
    
    analyze_experiment(exp_name, base_path)

if __name__ == "__main__":
    main()
