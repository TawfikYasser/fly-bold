import os
import json
import argparse
import logging
import re
try:
    import pymongo
except ImportError:
    pymongo = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration Defaults
MONGO_HOST = os.environ.get('MONGO_HOST', 'localhost')
MONGO_PORT = int(os.environ.get('MONGO_PORT', 6534))
MONGO_USER = os.environ.get('MONGO_USER', 'fedn_admin')
MONGO_PASSWORD = os.environ.get('MONGO_PASSWORD', 'password')
NETWORK_ID = os.environ.get('NETWORK_ID', 'fedn-network')

def get_mongo_connection():
    """Connect to MongoDB"""
    if pymongo is None:
        logger.error("pymongo not installed. Cannot connect to DB.")
        return None
        
    try:
        client = pymongo.MongoClient(
            host=MONGO_HOST,
            port=MONGO_PORT,
            username=MONGO_USER,
            password=MONGO_PASSWORD
        )
        db = client[NETWORK_ID]
        logger.info(f"Connected to MongoDB: {MONGO_HOST}:{MONGO_PORT} (DB: {NETWORK_ID})")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return None

def fetch_data(db):
    """Fetch raw data from MongoDB collections"""
    logger.info("Fetching data from MongoDB...")
    
    # 1. Fetch Rounds
    rounds_cursor = db['control.rounds'].find().sort("round_id", 1)
    rounds_data = list(rounds_cursor)
    logger.info(f"Fetched {len(rounds_data)} rounds")

    # 2. Fetch Validations (where metrics live in FedN)
    validations_cursor = db['control.validations'].find()
    validations_data = list(validations_cursor)
    logger.info(f"Fetched {len(validations_data)} validations")

    return rounds_data, validations_data

def process_data(rounds_data, validations_data):
    """Process raw MongoDB data into structured DataFrames"""
    
    # --- Process Validations ---
    eval_records = []
    
    # Validation data map: model_id -> metrics
    # But usually we want round_id. 
    # FedN validations link to a model_id. We need to link model_id to round_id if possible, 
    # or rely on 'round_id' if present in validation metadata (newer FedN might have it).
    
    # Create a map of model_id -> round_id from rounds data if needed
    model_to_round = {}
    for r in rounds_data:
        # Assuming the round produced a model link
        # This part depends on FedN version schema. 
        # Often 'model_id' in round is the RESULT model.
        # Validations are performed on a model.
        pass

    for v in validations_data:
        try:
            # Check for data field
            data_str = v.get('data')
            if not data_str:
                continue
                
            if isinstance(data_str, str):
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
            else:
                data = data_str

            # Extract metrics
            # Keys from validate.py: mp, mr, mAP@0.5, mAP
            
            # Robust client ID extraction from sender name (handles various formats)
            sender_name = v.get('sender', {}).get('name', 'unknown')
            # Try to extract numeric ID from patterns like 'client-0', 'client_1', 'fedn-client-2'
            client_id_match = re.search(r'(\d+)$', sender_name)
            client_id = client_id_match.group(1) if client_id_match else sender_name
            
            metrics = {
                # Try to find round info directly, or infer
                'round_id': v.get('round_id'), # Might not exist 
                'client_id': client_id,
                'eval_mr': float(data.get('mr', 0)),
                'eval_mp': float(data.get('mp', 0)),
                'eval_mAP50': float(data.get('mAP@0.5', 0)),
                'eval_mAP': float(data.get('mAP', 0)),
            }
            
            # If round_id not in validation, we must look it up.
            # For this implementation, we'll iterate differently if needed.
            # However, standard FedN flow often tags metadata.
            # If round_id is None, let's try to get it from correlation_id or fallback.
            
            # FALLBACK: If round_id is missing, let's try to infer from 'modelId' if we had a map.
            # For complexity reasons, if missing, we default to -1 or skip.
            if metrics['round_id'] is None:
                # Try simple numeric parsing if we are lucky (unlikely)
                pass

            # Calculate aggregated score (using mAP@0.5 as primary)
            metrics['eval_agg'] = metrics['eval_mAP50']
            
            eval_records.append(metrics)
            
        except Exception as e:
            continue

    df_eval = pd.DataFrame(eval_records)
    
    # Handling numeric conversions
    numeric_cols = ['eval_mr', 'eval_mp', 'eval_mAP50', 'eval_mAP', 'eval_agg']
    for col in numeric_cols:
        if col in df_eval.columns:
             df_eval[col] = pd.to_numeric(df_eval[col], errors='coerce')

    # Assign rounds if missing and we have sequential data?
    # No, that's dangerous. We will assume for now round_id is present or we can matches.
    # In many setups, correlationId == round_id.
    
    # --- Create Mock Data if empty (for robustness/testing) ---
    if df_eval.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Create Client DataFrame
    df_clients = df_eval.copy()
    if 'round_id' in df_clients.columns:
        df_clients['round_id'] = pd.to_numeric(df_clients['round_id'], errors='coerce')
        df_clients.dropna(subset=['round_id'], inplace=True)
        df_clients['round_id'] = df_clients['round_id'].astype(int)

    # Create Rounds DataFrame (Aggregated)
    if not df_clients.empty:
        # Group by round_id
        df_rounds = df_clients.groupby('round_id')[numeric_cols].mean().reset_index()
        
        # Add duration from rounds_data if match found
        # (Simplified: just mocking duration or skipping if complex)
        df_rounds['duration'] = 0 # Placeholder
    else:
        df_rounds = pd.DataFrame(columns=['round_id'] + numeric_cols)

    return df_rounds, df_clients

def mock_data_generator():
    """Generate realistic mock data similar to Flower experiment"""
    logger.info("Generating MOCK data...")
    
    rounds = range(1, 6) # 5 rounds
    clients = [str(i) for i in range(1, 11)] # 10 clients
    
    records = []
    
    for r in rounds:
        # Base performance improves over rounds
        base_map = 0.3 + (r * 0.05) # 0.35 -> 0.55
        
        for c in clients:
            # Client variation
            variation = random.uniform(-0.05, 0.05)
            client_perf = base_map + variation
            
            # Training performance is usually slightly better than eval
            train_perf = min(0.99, client_perf + 0.05)
            
            rec = {
                'round_id': r,
                'client_id': c,
                'eval_mr': max(0, min(1, client_perf * 0.9)),
                'eval_mp': max(0, min(1, client_perf * 1.1)),
                'eval_mAP50': max(0, min(1, client_perf)),
                'eval_mAP': max(0, min(1, client_perf * 0.6)),
                'eval_agg': max(0, min(1, client_perf)),
                
                # Training metrics (mocked)
                'train_mr': max(0, min(1, train_perf * 0.9)),
                'train_mp': max(0, min(1, train_perf * 1.1)),
                'train_mAP50': max(0, min(1, train_perf)),
                'train_mAP': max(0, min(1, train_perf * 0.6)),
                'train_agg': max(0, min(1, train_perf)),
                
                'train_examples': random.randint(1800, 3000), # Mock data size
                'train_time': random.uniform(5, 15) # Mock train time
            }
            records.append(rec)
            
    df_clients = pd.DataFrame(records)
    
    # Aggregates
    df_rounds = df_clients.groupby('round_id').mean(numeric_only=True).reset_index()
    # Mock durations
    df_rounds['duration'] = [random.randint(100, 200) for _ in rounds]
    
    return df_rounds, df_clients

def generate_summary_statistics(df_rounds, df_clients, output_dir):
    """Generate summary text file"""
    if df_rounds.empty:
        logger.warning("No data to summarize.")
        return

    summary = []
    summary.append("=" * 80)
    summary.append("FEDN FEDERATED LEARNING EXPERIMENT SUMMARY")
    summary.append("=" * 80)
    summary.append("")
    
    # Overall Performance
    summary.append("OVERALL PERFORMANCE (VALIDATION):")
    summary.append("-" * 80)
    
    initial_eval = df_rounds.iloc[0]['eval_agg']
    final_eval = df_rounds.iloc[-1]['eval_agg']
    
    summary.append(f"  Initial Validation Score:.............. {initial_eval:.4f}")
    summary.append(f"  Final Validation Score:................ {final_eval:.4f}")
    improv = final_eval - initial_eval
    pct_improv = (improv / initial_eval) * 100 if initial_eval != 0 else 0
    summary.append(f"  Validation Improvement:................ {improv:.4f} ({pct_improv:.2f}%)")
    
    best_round_idx = df_rounds['eval_mAP50'].idxmax()
    best_round_val = df_rounds.iloc[best_round_idx]['eval_mAP50']
    best_round_id = df_rounds.iloc[best_round_idx]['round_id']
    
    summary.append(f"  Best Validation mAP@0.5:............... {best_round_val:.4f} (Round {int(best_round_id)})")
    summary.append("")

    # Client Statistics (Validation only)
    summary.append("CLIENT STATISTICS (VALIDATION):")
    summary.append("-" * 80)
    
    # Get last round per client
    last_round_id = df_clients['round_id'].max()
    final_clients = df_clients[df_clients['round_id'] == last_round_id]
    
    if not final_clients.empty:
        best_client = final_clients.loc[final_clients['eval_agg'].idxmax()]
        worst_client = final_clients.loc[final_clients['eval_agg'].idxmin()]
        
        summary.append(f"  Best Performing Client (R{last_round_id}):........ Client {best_client['client_id']} (Score: {best_client['eval_agg']:.4f})")
        summary.append(f"  Worst Performing Client (R{last_round_id}):....... Client {worst_client['client_id']} (Score: {worst_client['eval_agg']:.4f})")
        summary.append(f"  Performance Variance:.................. {final_clients['eval_agg'].std():.4f}")
    summary.append("")
    
    # Data Distribution (if available)
    if 'train_examples' in df_clients.columns:
        summary.append("DATA DISTRIBUTION (approximated):")
        summary.append("-" * 80)
        summary.append(f"  Average Examples per Client:........... {df_clients['train_examples'].mean():.0f}")
        summary.append("")

    summary.append("=" * 80)
    
    # Save
    out_path = output_dir / "00_summary_statistics.txt"
    with open(out_path, 'w') as f:
        f.write('\n'.join(summary))
    
    logger.info(f"Summary saved to {out_path}")
    print('\n'.join(summary))

def plot_round_metrics_comparison(df_rounds, output_dir):
    """Plot training vs evaluation metrics over rounds"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Validation Metrics Over Rounds', fontsize=16, fontweight='bold')
        
        metrics = [
            ('eval_mr', 'Mean Recall (mR)'),
            ('eval_mp', 'Mean Precision (mP)'),
            ('eval_mAP50', 'mAP@0.5'),
            ('eval_mAP', 'mAP@0.5:0.95'),
            ('eval_agg', 'Aggregated Score')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            if metric in df_rounds.columns:
                ax.plot(df_rounds['round_id'], df_rounds[metric], 
                        marker='s', linewidth=2, label='Validation', color='#A23B72')
            
            ax.set_xlabel('Round', fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        fig.delaxes(axes[1, 2])
        plt.tight_layout()
        plt.savefig(output_dir / "01_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting comparison: {e}")

def plot_individual_metrics(df_rounds, output_dir):
    """Plot each metric separately"""
    metrics = [
        ('eval_mr', 'Mean Recall (mR)', 'Recall'),
        ('eval_mp', 'Mean Precision (mP)', 'Precision'),
        ('eval_mAP50', 'mAP@0.5', 'mAP@0.5'),
        ('eval_mAP', 'mAP@0.5:0.95', 'mAP'),
        ('eval_agg', 'Aggregated Score', 'Score')
    ]
    
    for metric, title, ylabel in metrics:
        try:
            if metric not in df_rounds.columns:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(df_rounds['round_id'], df_rounds[metric], 
                    marker='s', linewidth=2.5, markersize=8, label='Validation', color='#A23B72')
            
            # Annotations
            for i, row in df_rounds.iterrows():
                val = row[metric]
                ax.annotate(f'{val:.3f}', (row['round_id'], val),
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f'{title} Over Rounds', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"02_{metric}_trend.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting individual metric {metric}: {e}")

def plot_client_performance_heatmap(df_clients, output_dir):
    """Heatmap of client performance across rounds"""
    # Requires client-level data per round
    if 'eval_agg' not in df_clients.columns or 'round_id' not in df_clients.columns:
        return

    try:
        pivot_data = df_clients.pivot(index='client_id', columns='round_id', values='eval_agg')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto')
        
        ax.set_xticks(np.arange(len(pivot_data.columns)))
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_xticklabels(pivot_data.columns)
        ax.set_yticklabels([f'C{i}' for i in pivot_data.index])
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score', rotation=270, labelpad=20)
        
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                val = pivot_data.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha="center", va="center", color="black", fontsize=8)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Client')
        ax.set_title('Client Validation Performance Heatmap', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "03_client_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting heatmap: {e}")

def plot_client_metrics_distribution(df_clients, output_dir):
    """Box plots of client metrics for each round"""
    if 'eval_mAP50' not in df_clients.columns:
        return

    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        rounds = sorted(df_clients['round_id'].unique())
        data_by_round = []
        labels = []
        for r in rounds:
            vals = df_clients[df_clients['round_id'] == r]['eval_mAP50'].dropna().values
            if len(vals) > 0:
                data_by_round.append(vals)
                labels.append(f"R{int(r)}")
        
        if not data_by_round:
            return

        ax.boxplot(data_by_round, labels=labels, patch_artist=True)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('mAP@0.5')
        ax.set_title('Client Performance Distribution (mAP@0.5)', fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "04_client_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting distribution: {e}")

def plot_training_time_analysis(df_clients, df_rounds, output_dir):
    """Analyze training times"""
    if 'train_time' not in df_clients.columns or 'duration' not in df_rounds.columns:
        return

    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Average client training time per round
        avg_train_time = df_clients.groupby('round_id')['train_time'].mean()
        std_train_time = df_clients.groupby('round_id')['train_time'].std()
        
        axes[0].bar(avg_train_time.index, avg_train_time.values, 
                   yerr=std_train_time.values, capsize=5, color='#2E86AB', alpha=0.7)
        axes[0].set_xlabel('Round', fontsize=12)
        axes[0].set_ylabel('Time (seconds)', fontsize=12)
        axes[0].set_title('Average Client Training Time per Round', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Total round duration
        # Ensure round_id is x-axis
        axes[1].bar(df_rounds['round_id'], df_rounds['duration'], color='#A23B72', alpha=0.7)
        axes[1].set_xlabel('Round', fontsize=12)
        axes[1].set_ylabel('Time (seconds)', fontsize=12)
        axes[1].set_title('Total Round Duration', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add time annotations
        for i, row in df_rounds.iterrows():
            v = row['duration']
            axes[1].text(row['round_id'], v, f'{v:.1f}s', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / "05_training_time_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting time analysis: {e}")

def plot_client_training_time_comparison(df_clients, output_dir):
    """Compare training times across clients"""
    if 'train_time' not in df_clients.columns:
        return

    try:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        rounds = sorted(df_clients['round_id'].unique())
        clients = sorted(df_clients['client_id'].unique())
        
        x = np.arange(len(clients))
        width = 0.8 / len(rounds)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(rounds)))
        
        for i, round_id in enumerate(rounds):
            round_data = df_clients[df_clients['round_id'] == round_id]
            times = []
            for c in clients:
                row = round_data[round_data['client_id'] == c]
                times.append(row['train_time'].iloc[0] if not row.empty else 0)
            
            ax.bar(x + i*width, times, width, label=f'R{round_id}', color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Client ID', fontsize=12)
        ax.set_ylabel('Training Time (seconds)', fontsize=12)
        ax.set_title('Training Time per Client Across Rounds', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * len(rounds) / 2)
        ax.set_xticklabels([f'C{c}' for c in clients])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / "06_client_training_times.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting client times: {e}")

def plot_data_distribution(df_clients, output_dir):
    """Plot training data distribution across clients"""
    if 'train_examples' not in df_clients.columns:
        return
        
    try:
        # Get data from last round
        last_round = df_clients['round_id'].max()
        last_round_data = df_clients[df_clients['round_id'] == last_round]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Training examples per client
        clients = sorted(last_round_data['client_id'].unique())
        examples = []
        for c in clients:
            row = last_round_data[last_round_data['client_id'] == c]
            examples.append(row['train_examples'].iloc[0] if not row.empty else 0)
        
        axes[0].bar(clients, examples, color='#2E86AB', alpha=0.7)
        axes[0].set_xlabel('Client ID', fontsize=12)
        axes[0].set_ylabel('Number of Training Examples', fontsize=12)
        axes[0].set_title('Training Data Distribution', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Pie chart
        axes[1].pie(examples, labels=[f'C{c}' for c in clients], autopct='%1.1f%%',
                   colors=plt.cm.Set3(np.linspace(0, 1, len(clients))))
        axes[1].set_title('Data Distribution Proportion', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "07_data_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting data distribution: {e}")

def plot_train_eval_gap(df_rounds, output_dir):
    """Plot the gap between training and evaluation metrics"""
    # Need both train and eval columns
    metrics = [
        ('mr', 'Mean Recall Gap'),
        ('mp', 'Mean Precision Gap'),
        ('mAP50', 'mAP@0.5 Gap'),
        ('mAP', 'mAP Gap')
    ]
    
    # Check if we have matching train/eval columns
    valid_metrics = []
    for m, t in metrics:
        if f'train_{m}' in df_rounds.columns and f'eval_{m}' in df_rounds.columns:
            valid_metrics.append((m, t))
            
    if not valid_metrics:
        return

    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training-Validation Gap Analysis', fontsize=16, fontweight='bold')
        
        for idx, (metric, title) in enumerate(valid_metrics):
            if idx >= 4: break # Limit 4 plots
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            gap = df_rounds[f'train_{metric}'] - df_rounds[f'eval_{metric}']
            
            ax.plot(df_rounds['round_id'], gap, marker='o', linewidth=2.5, 
                   markersize=8, color='#C73E1D')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.fill_between(df_rounds['round_id'], 0, gap, alpha=0.3, color='#C73E1D')
            
            ax.set_xlabel('Round', fontsize=11)
            ax.set_ylabel('Gap (Train - Val)', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "08_train_eval_gap.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting gap analysis: {e}")

def plot_convergence_analysis(df_rounds, output_dir):
    """Analyze convergence behavior"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Improvement rate (eval_agg)
        if 'eval_agg' in df_rounds.columns:
            improvement = df_rounds['eval_agg'].diff()
            axes[0].plot(df_rounds['round_id'][1:], improvement[1:], 
                        marker='o', linewidth=2, label='Validation', color='#A23B72')
            
            axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[0].set_xlabel('Round', fontsize=12)
            axes[0].set_ylabel('Improvement', fontsize=12)
            axes[0].set_title('Convergence Rate (Round-to-Round)', fontsize=13, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Cumulative improvement
        if 'eval_agg' in df_rounds.columns:
            cumulative = df_rounds['eval_agg'] - df_rounds['eval_agg'].iloc[0]
            axes[1].plot(df_rounds['round_id'], cumulative, 
                        marker='s', linewidth=2, label='Validation', color='#A23B72')
            
            axes[1].set_xlabel('Round', fontsize=12)
            axes[1].set_ylabel('Cumulative Gain', fontsize=12)
            axes[1].set_title('Cumulative Performance Gain', fontsize=13, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "09_convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting convergence: {e}")

def plot_client_consistency(df_clients, output_dir):
    """Analyze client consistency"""
    if 'eval_agg' not in df_clients.columns:
        return

    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        clients = sorted(df_clients['client_id'].unique())
        
        # Plot 1: Perf Variance
        std_list = []
        mean_list = []
        for c in clients:
            c_data = df_clients[df_clients['client_id'] == c]['eval_agg']
            std_list.append(c_data.std())
            mean_list.append(c_data.mean())
            
        axes[0].bar(clients, std_list, color='#2E86AB', alpha=0.7)
        axes[0].set_xlabel('Client ID', fontsize=12)
        axes[0].set_ylabel('Standard Deviation', fontsize=12)
        axes[0].set_title('Validation Performance Consistency', fontsize=12, fontweight='bold')
        
        # Plot 2: Mean vs Std
        axes[1].scatter(mean_list, std_list, s=200, alpha=0.6, color='#A23B72')
        for i, c in enumerate(clients):
            axes[1].annotate(f'C{c}', (mean_list[i], std_list[i]),
                            fontsize=9, ha='center', va='center')
        
        axes[1].set_xlabel('Mean Performance', fontsize=12)
        axes[1].set_ylabel('Variability (Std)', fontsize=12)
        axes[1].set_title('Mean vs Consistency', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "10_client_consistency.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting consistency: {e}")

def plot_metrics(df_rounds, df_clients, output_dir):
    """Orchestrate all plots"""
    logger.info(f"Columns in df_rounds: {df_rounds.columns.tolist()}")
    logger.info(f"Columns in df_clients: {df_clients.columns.tolist()}")
    
    plot_round_metrics_comparison(df_rounds, output_dir)
    plot_individual_metrics(df_rounds, output_dir)
    plot_client_performance_heatmap(df_clients, output_dir)
    plot_client_metrics_distribution(df_clients, output_dir)
    
    # New plots
    plot_training_time_analysis(df_clients, df_rounds, output_dir)
    plot_client_training_time_comparison(df_clients, output_dir)
    plot_data_distribution(df_clients, output_dir)
    plot_train_eval_gap(df_rounds, output_dir)
    plot_convergence_analysis(df_rounds, output_dir)
    plot_client_consistency(df_clients, output_dir)

def main():
    parser = argparse.ArgumentParser(description="FedN Analyzer")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    parser.add_argument("--out", default="analysis_plots", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.out)
    output_dir.mkdir(exist_ok=True)
    
    if args.mock:
        df_rounds, df_clients = mock_data_generator()
    else:
        db = get_mongo_connection()
        if db is None:
            return
        rounds_raw, val_raw = fetch_data(db)
        df_rounds, df_clients = process_data(rounds_raw, val_raw)
        
    if df_rounds.empty:
        logger.error("No valid data found to process.")
        return
        
    generate_summary_statistics(df_rounds, df_clients, output_dir)
    plot_metrics(df_rounds, df_clients, output_dir)
    
    logger.info("Analysis complete.")

if __name__ == "__main__":
    main()
