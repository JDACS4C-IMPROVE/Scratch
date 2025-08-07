#!/usr/bin/env python3
"""
Enhanced Convergence Analysis Script for ML Training Runs
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

def calculate_adaptive_thresholds(all_final_losses):
    """Calculate performance thresholds based on actual data distribution"""
    losses = np.array([l for l in all_final_losses if l is not None])
    
    if len(losses) == 0:
        return {'excellent': 0, 'good': 0, 'poor': float('inf')}
    
    # Use percentiles for adaptive thresholds
    p10 = np.percentile(losses, 10)   # Top 10% - excellent
    p25 = np.percentile(losses, 25)   # Top 25% - good  
    p75 = np.percentile(losses, 75)   # Bottom 25% - poor
    
    return {
        'excellent': p10,
        'good': p25, 
        'poor': p75,
        'median': np.median(losses),
        'std': np.std(losses)
    }

def detect_training_completion_pattern(epoch_data):
    """Analyze if training completed naturally or was terminated early"""
    if len(epoch_data) < 3:
        return 'TOO_SHORT', {}
    
    val_losses = [ep['val_loss'] for ep in epoch_data if ep['val_loss'] is not None]
    
    if len(val_losses) < 3:
        return 'INSUFFICIENT_DATA', {}
    
    val_losses = np.array(val_losses)
    epochs = len(val_losses)
    
    # Calculate loss improvement patterns
    early_loss = np.mean(val_losses[:epochs//3]) if epochs >= 6 else val_losses[0]
    late_loss = np.mean(val_losses[-epochs//3:]) if epochs >= 6 else val_losses[-1]
    
    # Recent trend (last 30% of training)
    recent_portion = max(2, epochs//3)
    recent_losses = val_losses[-recent_portion:]
    recent_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0] if len(recent_losses) >= 2 else 0
    
    # Stability analysis
    recent_std = np.std(recent_losses)
    overall_improvement = (early_loss - late_loss) / early_loss if early_loss != 0 else 0
    
    # Pattern detection
    patterns = {
        'epochs': epochs,
        'overall_improvement': overall_improvement,
        'recent_trend': recent_trend,
        'recent_stability': recent_std,
        'early_loss': early_loss,
        'late_loss': late_loss
    }
    
    # Classification based on patterns
    if epochs < 5:
        return 'EARLY_TERMINATION', patterns
    elif abs(recent_trend) < recent_std * 0.1 and recent_std < abs(late_loss * 0.01):
        return 'NATURAL_CONVERGENCE', patterns
    elif recent_trend < -abs(late_loss * 0.001):  # Still decreasing significantly
        return 'STILL_IMPROVING', patterns  
    elif recent_trend > abs(late_loss * 0.001):   # Getting worse
        return 'DIVERGING', patterns
    else:
        return 'PLATEAUED', patterns

def classify_convergence_adaptive(epoch_data, thresholds):
    """Enhanced convergence classification with adaptive thresholds"""
    
    if len(epoch_data) < 2:
        return 'INSUFFICIENT_DATA', 0.0, {}
    
    # Extract valid losses
    val_losses = [ep['val_loss'] for ep in epoch_data if ep['val_loss'] is not None]
    epochs = [ep['epoch'] for ep in epoch_data if ep['val_loss'] is not None]
    
    if len(val_losses) < 2:
        return 'INSUFFICIENT_DATA', 0.0, {}
    
    val_losses = np.array(val_losses)
    epochs = np.array(epochs)
    
    # Basic metrics
    initial_loss = val_losses[0]
    final_loss = val_losses[-1]
    min_loss = np.min(val_losses)
    improvement = (initial_loss - final_loss) / initial_loss if initial_loss != 0 else 0
    
    # Trend analysis
    overall_slope = np.polyfit(epochs, val_losses, 1)[0] if len(val_losses) >= 2 else 0
    
    # Training completion analysis
    completion_pattern, pattern_info = detect_training_completion_pattern(epoch_data)
    
    # Performance tier based on adaptive thresholds
    if final_loss <= thresholds['excellent']:
        performance_tier = 'EXCELLENT'
    elif final_loss <= thresholds['good']:
        performance_tier = 'GOOD'
    elif final_loss <= thresholds['poor']:
        performance_tier = 'AVERAGE'
    else:
        performance_tier = 'POOR'
    
    metrics = {
        'initial_loss': float(initial_loss),
        'final_loss': float(final_loss),
        'min_loss': float(min_loss),
        'improvement_pct': float(improvement * 100),
        'overall_slope': float(overall_slope),
        'performance_tier': performance_tier,
        'completion_pattern': completion_pattern,
        'num_epochs': len(val_losses),
        'pattern_info': pattern_info
    }
    
    # Enhanced classification logic
    status, confidence = classify_by_performance_and_pattern(
        performance_tier, completion_pattern, improvement, pattern_info
    )
    
    return status, confidence, metrics

def classify_by_performance_and_pattern(performance_tier, completion_pattern, improvement, pattern_info):
    """Classify based on performance tier and training completion pattern"""
    
    # High confidence cases
    if performance_tier == 'EXCELLENT':
        if completion_pattern in ['NATURAL_CONVERGENCE', 'PLATEAUED']:
            return 'CONVERGED_EXCELLENT', 0.95
        elif completion_pattern == 'STILL_IMPROVING':
            return 'CONVERGING_EXCELLENT', 0.90
        else:
            return 'CONVERGED_EXCELLENT', 0.85  # Excellent performance regardless
    
    if performance_tier == 'GOOD':
        if completion_pattern == 'NATURAL_CONVERGENCE':
            return 'CONVERGED_GOOD', 0.90
        elif completion_pattern == 'STILL_IMPROVING':
            return 'CONVERGING_GOOD', 0.85
        elif completion_pattern == 'PLATEAUED':
            return 'CONVERGED_GOOD', 0.80
        elif completion_pattern == 'EARLY_TERMINATION' and improvement > 0.02:
            return 'INCOMPLETE_BUT_PROMISING', 0.70
        else:
            return 'CONVERGED_GOOD', 0.75
    
    if performance_tier == 'AVERAGE':
        if completion_pattern == 'STILL_IMPROVING' and improvement > 0.01:
            return 'CONVERGING_SLOWLY', 0.75
        elif completion_pattern in ['NATURAL_CONVERGENCE', 'PLATEAUED']:
            return 'CONVERGED_AVERAGE', 0.80
        elif completion_pattern == 'DIVERGING':
            return 'STALLED_OR_DIVERGING', 0.85
        else:
            return 'TRAINING_INCOMPLETE', 0.60
    
    # Poor performance cases
    if completion_pattern == 'DIVERGING':
        return 'DIVERGING', 0.90
    elif completion_pattern == 'STILL_IMPROVING' and improvement > 0.05:
        return 'SLOW_START_BUT_IMPROVING', 0.70
    elif completion_pattern == 'EARLY_TERMINATION':
        return 'FAILED_EARLY', 0.85
    else:
        return 'UNDERPERFORMING', 0.80

def analyze_experiment(exp_name, base_path=None):
    """Analyze all runs in an experiment with enhanced analytics"""
    
    if base_path is None:
        base_path = "/lus/flare/projects/candle_aesp_CNDA/out/unorun/Output"
    
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
    all_final_losses = []
    
    # First pass: collect all data and final losses for threshold calculation
    print("Collecting data...")
    for i, run_dir in enumerate(run_folders):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(run_folders)} runs")
            
        run_name = run_dir.name
        epoch_data = analyze_run(run_dir)
        
        if epoch_data:
            val_losses = [ep['val_loss'] for ep in epoch_data if ep['val_loss'] is not None]
            if val_losses:
                all_final_losses.append(val_losses[-1])
    
    # Calculate adaptive thresholds
    thresholds = calculate_adaptive_thresholds(all_final_losses)
    
    print(f"\nAdaptive Thresholds Calculated:")
    print(f"  Excellent (top 10%): ≤ {thresholds['excellent']:.6f}")
    print(f"  Good (top 25%):      ≤ {thresholds['good']:.6f}")
    print(f"  Poor (bottom 25%):   ≥ {thresholds['poor']:.6f}")
    print(f"  Median:              {thresholds['median']:.6f}")
    print()
    
    # Second pass: classify with adaptive thresholds
    print("Classifying convergence...")
    for i, run_dir in enumerate(run_folders):
        if i % 100 == 0:
            print(f"  Classified {i}/{len(run_folders)} runs")
            
        run_name = run_dir.name
        epoch_data = analyze_run(run_dir)
        
        if epoch_data is None:
            result = {
                'run': run_name,
                'status': 'NO_DATA',
                'confidence': 0.0,
                'num_epochs': 0,
                'final_loss': None,
                'improvement_pct': None,
                'performance_tier': 'NO_DATA'
            }
        else:
            status, confidence, metrics = classify_convergence_adaptive(epoch_data, thresholds)
            time_stats = calculate_time_stats(epoch_data)
            
            result = {
                'run': run_name,
                'status': status,
                'confidence': confidence,
                'num_epochs': metrics.get('num_epochs', 0),
                'final_loss': metrics.get('final_loss'),
                'improvement_pct': metrics.get('improvement_pct'),
                'performance_tier': metrics.get('performance_tier'),
                'completion_pattern': metrics.get('completion_pattern'),
                'avg_epoch_time_min': time_stats.get('avg_epoch_time'),
                'total_time_hrs': time_stats.get('total_time')
            }
            
            # Store detailed info
            detailed_result = result.copy()
            detailed_result.update(metrics)
            detailed_result.update(time_stats)
            detailed_result['epoch_data'] = epoch_data
            detailed_results.append(detailed_result)
        
        results.append(result)
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Generate reports
    generate_enhanced_summary(df, exp_name, thresholds)
    print("\n" + "="*80 + "\n")
    generate_actionable_report(detailed_results, exp_name)

def generate_enhanced_summary(df, exp_name, thresholds):
    """Generate comprehensive summary with actionable insights"""
    print("ENHANCED CONVERGENCE ANALYSIS")
    print("="*50)
    
    total_runs = len(df)
    
    # Status distribution
    status_counts = df['status'].value_counts()
    print(f"Total Runs Analyzed: {total_runs}")
    print(f"\nConvergence Status Distribution:")
    for status, count in status_counts.items():
        pct = (count / total_runs) * 100
        print(f"  {status:25}: {count:4d} ({pct:5.1f}%)")
    
    # Performance tier analysis
    valid_df = df[df['performance_tier'] != 'NO_DATA']
    if len(valid_df) > 0:
        print(f"\nPerformance Tier Analysis (n={len(valid_df)}):")
        tier_counts = valid_df['performance_tier'].value_counts()
        for tier, count in tier_counts.items():
            pct = (count / len(valid_df)) * 100
            print(f"  {tier:12}: {count:4d} ({pct:5.1f}%)")
    
    # Training completion patterns
    pattern_df = df[df['completion_pattern'].notna()]
    if len(pattern_df) > 0:
        print(f"\nTraining Completion Patterns:")
        pattern_counts = pattern_df['completion_pattern'].value_counts()
        for pattern, count in pattern_counts.items():
            pct = (count / len(pattern_df)) * 100
            print(f"  {pattern:20}: {count:4d} ({pct:5.1f}%)")
    
    # Performance metrics
    valid_metrics_df = df[df['final_loss'].notna()]
    if len(valid_metrics_df) > 0:
        print(f"\nPerformance Metrics (n={len(valid_metrics_df)}):")
        print(f"  Final Loss    - Mean: {valid_metrics_df['final_loss'].mean():.6f}, Std: {valid_metrics_df['final_loss'].std():.6f}")
        print(f"  Final Loss    - Min:  {valid_metrics_df['final_loss'].min():.6f}, Max: {valid_metrics_df['final_loss'].max():.6f}")
        
        improvement_valid = valid_metrics_df[valid_metrics_df['improvement_pct'].notna()]
        if len(improvement_valid) > 0:
            print(f"  Improvement % - Mean: {improvement_valid['improvement_pct'].mean():.1f}%, Std: {improvement_valid['improvement_pct'].std():.1f}%")
        
        epochs_valid = valid_metrics_df[valid_metrics_df['num_epochs'] > 0]
        if len(epochs_valid) > 0:
            print(f"  Epochs        - Mean: {epochs_valid['num_epochs'].mean():.1f}, Std: {epochs_valid['num_epochs'].std():.1f}")
    
    # Success rate calculation
    successful_statuses = ['CONVERGED_EXCELLENT', 'CONVERGED_GOOD', 'CONVERGING_EXCELLENT', 'CONVERGING_GOOD']
    successful_runs = df[df['status'].isin(successful_statuses)]
    success_rate = (len(successful_runs) / total_runs) * 100 if total_runs > 0 else 0
    
    print(f"\nOverall Success Rate: {success_rate:.1f}% ({len(successful_runs)}/{total_runs})")
    
    # Top performers
    excellent_runs = df[df['performance_tier'] == 'EXCELLENT'].nsmallest(5, 'final_loss')
    if len(excellent_runs) > 0:
        print(f"\nTop 5 Excellent Performers:")
        for _, run in excellent_runs.iterrows():
            print(f"  {run['run']}: Loss={run['final_loss']:.6f}, Status={run['status']}, Epochs={run['num_epochs']}")

def generate_actionable_report(detailed_results, exp_name, max_display=20):
    """Generate actionable insights and recommendations"""
    print("ACTIONABLE INSIGHTS & RECOMMENDATIONS")
    print("="*50)
    
    if not detailed_results:
        print("No detailed results available for analysis.")
        return
    
    # Sort by actionable priority: problems first, then best performers
    def priority_sort_key(x):
        status_priority = {
            'FAILED_EARLY': 0, 'DIVERGING': 1, 'UNDERPERFORMING': 2,
            'SLOW_START_BUT_IMPROVING': 3, 'TRAINING_INCOMPLETE': 4,
            'CONVERGING_SLOWLY': 5, 'INCOMPLETE_BUT_PROMISING': 6,
            'CONVERGED_AVERAGE': 7, 'CONVERGING_GOOD': 8, 'CONVERGED_GOOD': 9,
            'CONVERGING_EXCELLENT': 10, 'CONVERGED_EXCELLENT': 11
        }
        return (status_priority.get(x['status'], 12), x.get('final_loss', float('inf')))
    
    sorted_results = sorted(detailed_results, key=priority_sort_key)
    
    # Problem analysis
    problem_statuses = ['FAILED_EARLY', 'DIVERGING', 'UNDERPERFORMING']
    problem_runs = [r for r in sorted_results if r['status'] in problem_statuses]
    
    if problem_runs:
        print(f"🚨 PRIORITY ISSUES ({len(problem_runs)} runs need attention):")
        for result in problem_runs[:10]:
            epochs = result.get('num_epochs', 0)
            completion = result.get('completion_pattern', 'UNKNOWN')
            improvement = result.get('improvement_pct', 0)
            print(f"  {result['run']:8} | {result['status']:20} | {epochs:2d} epochs | {completion:15} | {improvement:+5.1f}%")
        if len(problem_runs) > 10:
            print(f"  ... and {len(problem_runs) - 10} more problem runs")
    
    # Promising runs that need more time
    promising_statuses = ['CONVERGING_EXCELLENT', 'CONVERGING_GOOD', 'INCOMPLETE_BUT_PROMISING']
    promising_runs = [r for r in sorted_results if r['status'] in promising_statuses]
    
    if promising_runs:
        print(f"\n⚡ PROMISING RUNS ({len(promising_runs)} could benefit from longer training):")
        for result in promising_runs[:10]:
            epochs = result.get('num_epochs', 0)
            improvement = result.get('improvement_pct', 0)
            final_loss = result.get('final_loss', 0)
            print(f"  {result['run']:8} | Loss={final_loss:.6f} | {epochs:2d} epochs | Improving {improvement:+5.1f}%")
    
    # Success stories
    success_statuses = ['CONVERGED_EXCELLENT', 'CONVERGED_GOOD']
    success_runs = [r for r in sorted_results if r['status'] in success_statuses]
    
    if success_runs:
        print(f"\n✅ SUCCESSFUL RUNS ({len(success_runs)} achieved good convergence):")
        for result in success_runs[:10]:
            epochs = result.get('num_epochs', 0) 
            improvement = result.get('improvement_pct', 0)
            final_loss = result.get('final_loss', 0)
            total_time = result.get('total_time_hrs', 0)
            time_str = f"{total_time:.1f}h" if total_time else "N/A"
            print(f"  {result['run']:8} | Loss={final_loss:.6f} | {epochs:2d} epochs | {improvement:+5.1f}% | {time_str}")
    
    print(f"\n📊 SUMMARY RECOMMENDATIONS:")
    
    total_runs = len(detailed_results)
    success_rate = len(success_runs) / total_runs * 100 if total_runs > 0 else 0
    
    if success_rate > 50:
        print(f"  ✅ Good overall performance ({success_rate:.1f}% success rate)")
    elif success_rate > 25:
        print(f"  ⚠️  Mixed results ({success_rate:.1f}% success rate) - review hyperparameters")
    else:
        print(f"  🚨 Poor overall performance ({success_rate:.1f}% success rate) - major changes needed")
    
    if len(problem_runs) > total_runs * 0.1:
        print(f"  🔧 High failure rate ({len(problem_runs)} runs) - check initialization/learning rate")
    
    if len(promising_runs) > total_runs * 0.2:
        print(f"  ⏰ Many runs need more time ({len(promising_runs)} still improving)")
    
    avg_epochs = np.mean([r.get('num_epochs', 0) for r in detailed_results])
    if avg_epochs < 10:
        print(f"  📈 Consider longer training (avg {avg_epochs:.1f} epochs may be insufficient)")

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
