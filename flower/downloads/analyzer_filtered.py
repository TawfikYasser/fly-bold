import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import re


# ==================== CONFIGURATION ====================
INPUT_FILE = "EXP_YOLOv5_s_detection_42_logs.json"

# Extract experiment number
match = re.search(r"_detection_(\d+)_", INPUT_FILE)
exp_id = match.group(1) if match else "unknown"
OUTPUT_DIR = f"analysis_exp_{exp_id}_filtered"

# Experiment configuration (update based on your setup)
CONFIG = {
    "Experiment ID": exp_id,
    "Train Images/Client": 1000,
    "Val Images/Client": 500,
    "Total Clients": 10,
    "Server Rounds": 5,
    "Local Epochs": 3,
    "Batch Size": 32,
    "Learning Rate": 0.005,
    "YOLO Model": "s",
    "Image Size": 512,
    "Dirichlet Alpha": 0.7,
}


# ==================== DATA LOADING ====================
def load_data(filename):
    """Load JSON data from file"""
    with open(filename, 'r') as f:
        return json.load(f)


def extract_round_metrics(data):
    """Extract round-level metrics (both training and evaluation)"""
    rounds = []
    for round_data in data:
        round_dict = {
            'round_id': round_data['round_id'],
            'duration': round_data['round_duration'],
            'lr': round_data['lr'],
            # Training metrics
            'train_loss': round_data.get('round_train_loss', 0.0),
            'train_mr': round_data['round_training_acc']['mr'],
            'train_mp': round_data['round_training_acc']['mp'],
            'train_mAP50': round_data['round_training_acc']['mAP@0.5'],
            'train_mAP': round_data['round_training_acc']['mAP'],
            'train_agg': round_data['round_training_acc']['aggregated'],
            # Evaluation metrics
            'eval_mr': round_data['round_eval_acc']['mr'],
            'eval_mp': round_data['round_eval_acc']['mp'],
            'eval_mAP50': round_data['round_eval_acc']['mAP@0.5'],
            'eval_mAP': round_data['round_eval_acc']['mAP'],
            'eval_agg': round_data['round_eval_acc']['aggregated'],
            'eval_loss': round_data.get('round_eval_loss', 0.0),
            'eval_time': round_data.get('round_eval_time', 0),
            'data_mb': round_data.get('round_data_transferred_mb', 0),
        }
        rounds.append(round_dict)
    return pd.DataFrame(rounds)


def extract_client_metrics(data):
    """Extract per-client metrics across all rounds"""
    client_data = []
    for round_data in data:
        round_id = round_data['round_id']
        for client in round_data['clients_logs']:
            client_info = {
                'round_id': round_id,
                'client_id': client['client_id'],
                'train_time': client['client_train_time'],
                'train_examples': client['client_train_num_examples'],
                'eval_time': client.get('client_eval_time', 0),
                'eval_examples': client.get('client_eval_num_examples', 0),
                # Training metrics
                'train_loss': client.get('client_train_loss', 0.0),
                'train_mr': client['client_train_acc']['mr'],
                'train_mp': client['client_train_acc']['mp'],
                'train_mAP50': client['client_train_acc']['mAP@0.5'],
                'train_mAP': client['client_train_acc']['mAP'],
                'train_agg': client['client_train_acc']['aggregated'],
                # Evaluation metrics
                'eval_mr': client.get('client_eval_acc', {}).get('mr', 0.0),
                'eval_mp': client.get('client_eval_acc', {}).get('mp', 0.0),
                'eval_mAP50': client.get('client_eval_acc', {}).get('mAP@0.5', 0.0),
                'eval_mAP': client.get('client_eval_acc', {}).get('mAP', 0.0),
                'eval_agg': client.get('client_eval_acc', {}).get('aggregated', 0.0),
                'eval_loss': client.get('client_eval_loss', 0.0),
            }
            client_data.append(client_info)
    return pd.DataFrame(client_data)


def filter_zero_metrics_clients(df_clients):
    """
    Filter out clients with zero metrics in training or validation.
    Excludes clients that have any zero values in their train or eval metrics.
    """
    # Define metric columns to check
    train_metrics = ['train_mr', 'train_mp', 'train_mAP50', 'train_mAP', 'train_agg']
    eval_metrics = ['eval_mr', 'eval_mp', 'eval_mAP50', 'eval_mAP', 'eval_agg']
    
    # Get clients with any zero values in train or eval metrics
    zero_train = (df_clients[train_metrics] == 0).any(axis=1)
    zero_eval = (df_clients[eval_metrics] == 0).any(axis=1)
    
    # Identify clients to remove (those with zero metrics in either train or eval)
    clients_to_remove = df_clients[zero_train | zero_eval]['client_id'].unique()
    
    # Filter out these clients
    df_clients_filtered = df_clients[~df_clients['client_id'].isin(clients_to_remove)].copy()
    
    # Log filtering info
    print(f"\n[*] Client Filtering Summary:")
    print(f"    • Original clients: {df_clients['client_id'].nunique()}")
    print(f"    • Clients removed (zero metrics): {len(clients_to_remove)}")
    if len(clients_to_remove) > 0:
        print(f"    • Removed client IDs: {sorted(clients_to_remove)}")
    print(f"    • Remaining clients: {df_clients_filtered['client_id'].nunique()}")
    print(f"    • Original records: {len(df_clients)}")
    print(f"    • Filtered records: {len(df_clients_filtered)}")
    
    return df_clients_filtered


# ==================== PLOTTING FUNCTIONS ====================

def plot_train_vs_eval_metrics(df_rounds, output_dir):
    """Plot training vs evaluation metrics side by side"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training vs Validation Performance Over Rounds', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    metrics = [
        ('mr', 'Mean Recall (mR)'),
        ('mp', 'Mean Precision (mP)'),
        ('mAP50', 'mAP@0.5'),
        ('mAP', 'mAP@0.5:0.95'),
        ('agg', 'Aggregated Score'),
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        # Plot both train and eval
        ax.plot(df_rounds['round_id'], df_rounds[f'train_{metric}'], 
                marker='o', linewidth=2.5, markersize=8, label='Training', 
                color='#2E86AB', alpha=0.8)
        ax.plot(df_rounds['round_id'], df_rounds[f'eval_{metric}'], 
                marker='s', linewidth=2.5, markersize=8, label='Validation', 
                color='#A23B72', alpha=0.8)
        
        # Fill area between for visualization
        ax.fill_between(df_rounds['round_id'], 
                        df_rounds[f'train_{metric}'], 
                        df_rounds[f'eval_{metric}'],
                        alpha=0.2, color='gray')
        
        # Annotations on last point
        last_train = df_rounds[f'train_{metric}'].iloc[-1]
        last_eval = df_rounds[f'eval_{metric}'].iloc[-1]
        last_round = df_rounds['round_id'].iloc[-1]
        
        ax.annotate(f'{last_train:.3f}', 
                   (last_round, last_train),
                   textcoords="offset points", xytext=(5, 5), 
                   ha='left', fontsize=9, fontweight='bold', color='#2E86AB')
        ax.annotate(f'{last_eval:.3f}', 
                   (last_round, last_eval),
                   textcoords="offset points", xytext=(5, -12), 
                   ha='left', fontsize=9, fontweight='bold', color='#A23B72')
        
        ax.set_xlabel('Round', fontsize=11, fontweight='bold')
        ax.set_ylabel(title.split('(')[0].strip(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(df_rounds['round_id'])
    
    # Loss comparison in the 6th subplot
    fig.delaxes(axes[1, 2])
    ax_loss = fig.add_subplot(2, 3, 6)
    
    ax_loss.plot(df_rounds['round_id'], df_rounds['train_loss'], 
                marker='o', linewidth=2.5, markersize=8, label='Training', 
                color='#C73E1D', alpha=0.8)
    ax_loss.plot(df_rounds['round_id'], df_rounds['eval_loss'], 
                marker='s', linewidth=2.5, markersize=8, label='Validation', 
                color='#F18F01', alpha=0.8)
    
    ax_loss.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax_loss.set_ylabel('Loss', fontsize=11)
    ax_loss.set_title('Training vs Validation Loss', fontsize=12, fontweight='bold', pad=10)
    ax_loss.legend(loc='best', fontsize=10)
    ax_loss.grid(True, alpha=0.3, linestyle='--')
    ax_loss.set_xticks(df_rounds['round_id'])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_train_vs_eval_metrics.png", 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_generalization_gap(df_rounds, output_dir):
    """Plot the generalization gap (train - eval) for all metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Generalization Gap Analysis (Training - Validation)', 
                 fontsize=16, fontweight='bold')
    
    metrics = [
        ('mr', 'Mean Recall Gap'),
        ('mp', 'Mean Precision Gap'),
        ('mAP50', 'mAP@0.5 Gap'),
        ('mAP', 'mAP Gap'),
        ('agg', 'Aggregated Score Gap'),
        ('loss', 'Loss Gap (Train - Val)')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        gap = df_rounds[f'train_{metric}'] - df_rounds[f'eval_{metric}']
        
        # Color based on whether gap is positive (overfitting) or negative
        colors = ['#C73E1D' if g > 0 else '#6A994E' for g in gap]
        
        bars = ax.bar(df_rounds['round_id'], gap, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
        
        # Add value annotations
        for i, (r, g) in enumerate(zip(df_rounds['round_id'], gap)):
            ax.text(r, g, f'{g:.3f}', 
                   ha='center', va='bottom' if g > 0 else 'top', 
                   fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Round', fontsize=11, fontweight='bold')
        ax.set_ylabel('Gap', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(df_rounds['round_id'])
        
        # Add interpretation text
        avg_gap = gap.mean()
        status = "Overfitting" if avg_gap > 0.02 else "Good Generalization"
        color = '#C73E1D' if avg_gap > 0.02 else '#6A994E'
        ax.text(0.02, 0.98, f'Avg: {avg_gap:.3f}\n{status}', 
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_generalization_gap.png", 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_client_performance_heatmap(df_clients, output_dir):
    """Heatmap showing each client's performance across rounds"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Client Performance Evolution', 
                 fontsize=15, fontweight='bold')
    
    metrics_data = [
        ('train_agg', 'Training Aggregated Score', axes[0, 0]),
        ('eval_agg', 'Validation Aggregated Score', axes[0, 1]),
        ('train_mAP50', 'Training mAP@0.5', axes[1, 0]),
        ('eval_mAP50', 'Validation mAP@0.5', axes[1, 1])
    ]
    
    for metric, title, ax in metrics_data:
        pivot = df_clients.pivot(index='client_id', columns='round_id', values=metric)
        
        # Create heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', 
                      vmin=pivot.values.min()*0.95, vmax=pivot.values.max()*1.02)
        
        # Set ticks
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([f'R{c}' for c in pivot.columns], fontsize=10)
        ax.set_yticklabels([f'C{i}' for i in pivot.index], fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(title, rotation=270, labelpad=20, fontsize=11)
        
        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                if not np.isnan(value):
                    text_color = 'white' if value < pivot.values.mean() else 'black'
                    ax.text(j, i, f'{value:.3f}',
                           ha="center", va="center", color=text_color, 
                           fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Client ID', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_client_performance_heatmap.png", 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_client_train_vs_eval_comparison(df_clients, output_dir):
    """Compare client training vs evaluation performance"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Client Training vs Validation Performance', 
                 fontsize=15, fontweight='bold')
    
    # Last round data
    last_round = df_clients[df_clients['round_id'] == df_clients['round_id'].max()]
    clients = sorted(last_round['client_id'].unique())
    
    # 1. Aggregated score comparison
    ax = axes[0, 0]
    x = np.arange(len(clients))
    width = 0.35
    
    train_scores = [last_round[last_round['client_id'] == c]['train_agg'].values[0] for c in clients]
    eval_scores = [last_round[last_round['client_id'] == c]['eval_agg'].values[0] for c in clients]
    
    ax.bar(x - width/2, train_scores, width, label='Training', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, eval_scores, width, label='Validation', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Client ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Aggregated Score', fontsize=11)
    ax.set_title('Final Round: Training vs Validation Score', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{c}' for c in clients])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. mAP@0.5 comparison
    ax = axes[0, 1]
    train_map = [last_round[last_round['client_id'] == c]['train_mAP50'].values[0] for c in clients]
    eval_map = [last_round[last_round['client_id'] == c]['eval_mAP50'].values[0] for c in clients]
    
    ax.bar(x - width/2, train_map, width, label='Training', color='#F18F01', alpha=0.8)
    ax.bar(x + width/2, eval_map, width, label='Validation', color='#C73E1D', alpha=0.8)
    
    ax.set_xlabel('Client ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('mAP@0.5', fontsize=11)
    ax.set_title('Final Round: Training vs Validation mAP@0.5', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{c}' for c in clients])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Scatter: Training vs Validation performance
    ax = axes[1, 0]
    ax.scatter(train_scores, eval_scores, s=200, alpha=0.6, 
              c=range(len(clients)), cmap='viridis', edgecolors='black', linewidth=2)
    
    # Add diagonal line (perfect generalization)
    min_val = min(min(train_scores), min(eval_scores))
    max_val = max(max(train_scores), max(eval_scores))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.5, label='Perfect Generalization')
    
    for i, c in enumerate(clients):
        ax.annotate(f'C{c}', (train_scores[i], eval_scores[i]),
                   fontsize=10, ha='center', va='center', fontweight='bold', color='white')
    
    ax.set_xlabel('Training Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Score', fontsize=12, fontweight='bold')
    ax.set_title('Training vs Validation Correlation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Generalization gap per client
    ax = axes[1, 1]
    gaps = [train_scores[i] - eval_scores[i] for i in range(len(clients))]
    colors = ['#C73E1D' if g > 0 else '#6A994E' for g in gaps]
    
    ax.bar(x, gaps, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    
    for i, (c, g) in enumerate(zip(clients, gaps)):
        ax.text(i, g, f'{g:.3f}', ha='center', 
               va='bottom' if g > 0 else 'top', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Client ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gap (Train - Val)', fontsize=11)
    ax.set_title('Generalization Gap per Client', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{c}' for c in clients])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_client_train_vs_eval.png", 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_time_analysis(df_clients, df_rounds, output_dir):
    """Comprehensive time analysis"""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Training Time Analysis', fontsize=16, fontweight='bold')
    
    # 1. Round duration over time
    ax1 = fig.add_subplot(gs[0, :])
    rounds = df_rounds['round_id'].values
    duration_min = df_rounds['duration'].values / 60
    
    ax1.bar(rounds, duration_min, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.plot(rounds, duration_min, 'ro-', linewidth=2, markersize=8)
    
    for i, (r, d) in enumerate(zip(rounds, duration_min)):
        ax1.text(r, d, f'{d:.1f}m', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Duration (minutes)', fontsize=12)
    ax1.set_title('Round Duration (Slowest Client)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(rounds)
    
    # 2. Average client training time per round
    ax2 = fig.add_subplot(gs[1, 0])
    avg_train = df_clients.groupby('round_id')['train_time'].mean() / 60
    std_train = df_clients.groupby('round_id')['train_time'].std() / 60
    
    ax2.bar(avg_train.index, avg_train.values, yerr=std_train.values, 
           capsize=5, color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Time (minutes)', fontsize=11)
    ax2.set_title('Avg Client Training Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(avg_train.index)
    
    # 3. Average client eval time per round
    ax3 = fig.add_subplot(gs[1, 1])
    avg_eval = df_clients.groupby('round_id')['eval_time'].mean()
    std_eval = df_clients.groupby('round_id')['eval_time'].std()
    
    ax3.bar(avg_eval.index, avg_eval.values, yerr=std_eval.values, 
           capsize=5, color='#F18F01', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Time (seconds)', fontsize=11)
    ax3.set_title('Avg Client Eval Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(avg_eval.index)
    
    # 4. Time breakdown pie chart (last round)
    ax4 = fig.add_subplot(gs[1, 2])
    last_round = df_clients[df_clients['round_id'] == df_clients['round_id'].max()]
    total_train = last_round['train_time'].sum()
    total_eval = last_round['eval_time'].sum()
    
    ax4.pie([total_train, total_eval], 
           labels=['Training', 'Evaluation'],
           autopct='%1.1f%%', startangle=90,
           colors=['#A23B72', '#F18F01'])
    ax4.set_title(f'Time Distribution (Round {last_round["round_id"].iloc[0]})', 
                 fontsize=12, fontweight='bold')
    
    # 5. Client training time comparison
    ax5 = fig.add_subplot(gs[2, :])
    clients = sorted(df_clients['client_id'].unique())
    rounds_list = sorted(df_clients['round_id'].unique())
    
    x = np.arange(len(clients))
    width = 0.15
    colors_palette = plt.cm.Set3(np.linspace(0, 1, len(rounds_list)))
    
    for i, round_id in enumerate(rounds_list):
        round_data = df_clients[df_clients['round_id'] == round_id]
        times = []
        for c in clients:
            row = round_data[round_data['client_id'] == c]
            times.append(row['train_time'].iloc[0] / 60 if not row.empty else 0)
        
        ax5.bar(x + i*width, times, width, label=f'Round {round_id}', 
               color=colors_palette[i], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax5.set_xlabel('Client ID', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Training Time (minutes)', fontsize=12)
    ax5.set_title('Training Time per Client Across Rounds', fontsize=13, fontweight='bold')
    ax5.set_xticks(x + width * (len(rounds_list)-1) / 2)
    ax5.set_xticklabels([f'C{c}' for c in clients])
    ax5.legend(loc='upper right', fontsize=9, ncol=len(rounds_list))
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(f"{output_dir}/05_time_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_convergence_analysis(df_rounds, df_clients, output_dir):
    """Convergence behavior analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Convergence Analysis', fontsize=15, fontweight='bold')
    
    # 1. Training improvement rate
    ax = axes[0, 0]
    train_improvement = df_rounds['train_agg'].diff()
    eval_improvement = df_rounds['eval_agg'].diff()
    
    ax.plot(df_rounds['round_id'][1:], train_improvement[1:], 
            marker='o', linewidth=2, label='Training', color='#2E86AB')
    ax.plot(df_rounds['round_id'][1:], eval_improvement[1:], 
            marker='s', linewidth=2, label='Validation', color='#A23B72')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax.set_ylabel('Improvement from Previous Round', fontsize=11)
    ax.set_title('Round-to-Round Improvement', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df_rounds['round_id'][1:])
    
    # 2. Cumulative improvement
    ax = axes[0, 1]
    train_cumulative = df_rounds['train_agg'] - df_rounds['train_agg'].iloc[0]
    eval_cumulative = df_rounds['eval_agg'] - df_rounds['eval_agg'].iloc[0]
    
    ax.plot(df_rounds['round_id'], train_cumulative, 
            marker='o', linewidth=2.5, label='Training', color='#2E86AB')
    ax.plot(df_rounds['round_id'], eval_cumulative, 
            marker='s', linewidth=2.5, label='Validation', color='#A23B72')
    
    ax.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Improvement', fontsize=11)
    ax.set_title('Total Performance Gain', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df_rounds['round_id'])
    
    # 3. Training loss convergence
    ax = axes[0, 2]
    ax.plot(df_rounds['round_id'], df_rounds['train_loss'], 
            marker='o', linewidth=2.5, markersize=8, color='#C73E1D', label='Training')
    ax.plot(df_rounds['round_id'], df_rounds['eval_loss'], 
            marker='s', linewidth=2.5, markersize=8, color='#F18F01', label='Validation')
    
    ax.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Loss Convergence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df_rounds['round_id'])
    
    # 4. Client training variance
    ax = axes[1, 0]
    train_var = df_clients.groupby('round_id')['train_agg'].std()
    eval_var = df_clients.groupby('round_id')['eval_agg'].std()
    
    ax.plot(train_var.index, train_var.values, 
            marker='o', linewidth=3, markersize=10, color='#2E86AB', label='Training')
    ax.plot(eval_var.index, eval_var.values, 
            marker='s', linewidth=3, markersize=10, color='#A23B72', label='Validation')
    
    ax.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax.set_ylabel('Client Performance Std Dev', fontsize=11)
    ax.set_title('Client Convergence (Lower = More Aligned)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(train_var.index)
    
    # 5. Learning curve with confidence interval
    ax = axes[1, 1]
    train_mean = df_clients.groupby('round_id')['train_agg'].mean()
    train_std = df_clients.groupby('round_id')['train_agg'].std()
    eval_mean = df_clients.groupby('round_id')['eval_agg'].mean()
    eval_std = df_clients.groupby('round_id')['eval_agg'].std()
    
    ax.plot(train_mean.index, train_mean.values, 
            'o-', linewidth=3, markersize=10, color='#2E86AB', label='Train Mean')
    ax.fill_between(train_mean.index, 
                     train_mean.values - train_std.values,
                     train_mean.values + train_std.values,
                     alpha=0.2, color='#2E86AB')
    
    ax.plot(eval_mean.index, eval_mean.values, 
            's-', linewidth=3, markersize=10, color='#A23B72', label='Val Mean')
    ax.fill_between(eval_mean.index, 
                     eval_mean.values - eval_std.values,
                     eval_mean.values + eval_std.values,
                     alpha=0.2, color='#A23B72')
    
    ax.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax.set_ylabel('Performance', fontsize=11)
    ax.set_title('Learning Curves with Uncertainty', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(train_mean.index)
    
    # 6. Overfitting indicator
    ax = axes[1, 2]
    gap = df_rounds['train_agg'] - df_rounds['eval_agg']
    
    colors = ['#C73E1D' if g > 0.02 else '#6A994E' for g in gap]
    ax.bar(df_rounds['round_id'], gap, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=0.02, color='red', linestyle=':', alpha=0.5, label='Overfitting Threshold')
    
    for r, g in zip(df_rounds['round_id'], gap):
        ax.text(r, g, f'{g:.3f}', ha='center', 
               va='bottom' if g > 0 else 'top', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax.set_ylabel('Gap (Train - Val)', fontsize=11)
    ax.set_title('Overfitting Indicator', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(df_rounds['round_id'])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/06_convergence_analysis.png", 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_communication_overhead(df_rounds, output_dir):
    """Communication overhead analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Communication Overhead', fontsize=15, fontweight='bold')
    
    # 1. Data transferred per round
    ax1 = axes[0]
    ax1.bar(df_rounds['round_id'], df_rounds['data_mb'], 
           color='#6A994E', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for i, (r, d) in enumerate(zip(df_rounds['round_id'], df_rounds['data_mb'])):
        ax1.text(r, d, f'{d:.1f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Data Transferred (MB)', fontsize=12)
    ax1.set_title('Communication per Round', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(df_rounds['round_id'])
    
    # 2. Cumulative data
    ax2 = axes[1]
    cumulative = df_rounds['data_mb'].cumsum()
    
    ax2.plot(df_rounds['round_id'], cumulative, marker='o', 
            linewidth=3, markersize=10, color='#6A994E')
    ax2.fill_between(df_rounds['round_id'], 0, cumulative, 
                     alpha=0.3, color='#6A994E')
    
    for i, (r, c) in enumerate(zip(df_rounds['round_id'], cumulative)):
        ax2.annotate(f'{c:.1f} MB', (r, c),
                    textcoords="offset points", xytext=(0, 10), 
                    ha='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Data (MB)', fontsize=12)
    ax2.set_title(f'Total: {cumulative.iloc[-1]:.1f} MB ({cumulative.iloc[-1]/1024:.2f} GB)', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(df_rounds['round_id'])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_communication_overhead.png", 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_client_analysis(df_clients, output_dir):
    """Generate individual plots for each client showing train/val accuracy and round times"""
    clients = sorted(df_clients['client_id'].unique())
    
    for client_id in clients:
        # Filter data for this client
        client_data = df_clients[df_clients['client_id'] == client_id].sort_values('round_id')
        
        if client_data.empty:
            continue
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Client {client_id} Performance Analysis', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        rounds = client_data['round_id'].values
        
        # 1. Train vs Val Aggregated Score
        ax = axes[0, 0]
        ax.plot(rounds, client_data['train_agg'], 
                marker='o', linewidth=2.5, markersize=8, label='Training', 
                color='#2E86AB', alpha=0.8)
        ax.plot(rounds, client_data['eval_agg'], 
                marker='s', linewidth=2.5, markersize=8, label='Validation', 
                color='#A23B72', alpha=0.8)
        
        # Fill area between
        ax.fill_between(rounds, client_data['train_agg'], client_data['eval_agg'],
                        alpha=0.2, color='gray')
        
        # Annotations on last point
        last_train_agg = client_data['train_agg'].iloc[-1]
        last_eval_agg = client_data['eval_agg'].iloc[-1]
        last_round = rounds[-1]
        
        ax.annotate(f'{last_train_agg:.3f}', (last_round, last_train_agg),
                   textcoords="offset points", xytext=(5, 5), 
                   ha='left', fontsize=9, fontweight='bold', color='#2E86AB')
        ax.annotate(f'{last_eval_agg:.3f}', (last_round, last_eval_agg),
                   textcoords="offset points", xytext=(5, -12), 
                   ha='left', fontsize=9, fontweight='bold', color='#A23B72')
        
        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Aggregated Score (Train vs Val)', fontsize=13, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(rounds)
        
        # 2. Train vs Val mAP@0.5
        ax = axes[0, 1]
        ax.plot(rounds, client_data['train_mAP50'], 
                marker='o', linewidth=2.5, markersize=8, label='Training', 
                color='#F18F01', alpha=0.8)
        ax.plot(rounds, client_data['eval_mAP50'], 
                marker='s', linewidth=2.5, markersize=8, label='Validation', 
                color='#C73E1D', alpha=0.8)
        
        # Fill area between
        ax.fill_between(rounds, client_data['train_mAP50'], client_data['eval_mAP50'],
                        alpha=0.2, color='gray')
        
        # Annotations on last point
        last_train_map50 = client_data['train_mAP50'].iloc[-1]
        last_eval_map50 = client_data['eval_mAP50'].iloc[-1]
        
        ax.annotate(f'{last_train_map50:.3f}', (last_round, last_train_map50),
                   textcoords="offset points", xytext=(5, 5), 
                   ha='left', fontsize=9, fontweight='bold', color='#F18F01')
        ax.annotate(f'{last_eval_map50:.3f}', (last_round, last_eval_map50),
                   textcoords="offset points", xytext=(5, -12), 
                   ha='left', fontsize=9, fontweight='bold', color='#C73E1D')
        
        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('mAP@0.5', fontsize=11)
        ax.set_title('mAP@0.5 (Train vs Val)', fontsize=13, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(rounds)
        
        # 3. Train vs Val mAP@0.5:0.95
        ax = axes[1, 0]
        ax.plot(rounds, client_data['train_mAP'], 
                marker='o', linewidth=2.5, markersize=8, label='Training', 
                color='#6A994E', alpha=0.8)
        ax.plot(rounds, client_data['eval_mAP'], 
                marker='s', linewidth=2.5, markersize=8, label='Validation', 
                color='#BC4B51', alpha=0.8)
        
        # Fill area between
        ax.fill_between(rounds, client_data['train_mAP'], client_data['eval_mAP'],
                        alpha=0.2, color='gray')
        
        # Annotations on last point
        last_train_map = client_data['train_mAP'].iloc[-1]
        last_eval_map = client_data['eval_mAP'].iloc[-1]
        
        ax.annotate(f'{last_train_map:.3f}', (last_round, last_train_map),
                   textcoords="offset points", xytext=(5, 5), 
                   ha='left', fontsize=9, fontweight='bold', color='#6A994E')
        ax.annotate(f'{last_eval_map:.3f}', (last_round, last_eval_map),
                   textcoords="offset points", xytext=(5, -12), 
                   ha='left', fontsize=9, fontweight='bold', color='#BC4B51')
        
        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('mAP@0.5:0.95', fontsize=11)
        ax.set_title('mAP@0.5:0.95 (Train vs Val)', fontsize=13, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(rounds)
        
        # 4. Training Time per Round
        ax = axes[1, 1]
        train_time_min = client_data['train_time'] / 60
        
        bars = ax.bar(rounds, train_time_min, color='#2E86AB', alpha=0.7, 
                     edgecolor='black', linewidth=1.5)
        ax.plot(rounds, train_time_min, 'ro-', linewidth=2, markersize=8)
        
        # Add value annotations on bars
        for round_id, time_val in zip(rounds, train_time_min):
            ax.text(round_id, time_val, f'{time_val:.1f}m', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Training Time (minutes)', fontsize=11)
        ax.set_title('Training Time per Round', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(rounds)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/08_client_{client_id}_analysis.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()


def generate_summary_report(df_rounds, df_clients, output_dir):
    """Generate comprehensive text summary"""
    summary = []
    
    summary.append("=" * 90)
    summary.append("FEDERATED LEARNING EXPERIMENT REPORT")
    summary.append("=" * 90)
    summary.append("")
    
    # Configuration
    summary.append("EXPERIMENT CONFIGURATION")
    summary.append("-" * 90)
    for key, value in CONFIG.items():
        summary.append(f"  {key:.<45} {value}")
    summary.append("")
    
    # Overall performance - TRAINING
    summary.append("TRAINING PERFORMANCE")
    summary.append("-" * 90)
    train_initial = df_rounds.iloc[0]['train_agg']
    train_final = df_rounds.iloc[-1]['train_agg']
    train_best = df_rounds['train_agg'].max()
    train_best_round = df_rounds['train_agg'].idxmax()
    
    summary.append(f"  Initial Training Score:.................... {train_initial:.4f}")
    summary.append(f"  Final Training Score:...................... {train_final:.4f}")
    summary.append(f"  Best Training Score:....................... {train_best:.4f} (Round {train_best_round})")
    summary.append(f"  Training Improvement:...................... {train_final - train_initial:.4f} ({(train_final-train_initial)/train_initial*100:+.2f}%)")
    summary.append("")
    summary.append(f"  Best Training mAP@0.5:..................... {df_rounds['train_mAP50'].max():.4f} (Round {df_rounds['train_mAP50'].idxmax()})")
    summary.append(f"  Best Training mAP:......................... {df_rounds['train_mAP'].max():.4f} (Round {df_rounds['train_mAP'].idxmax()})")
    summary.append(f"  Final Training Loss:....................... {df_rounds['train_loss'].iloc[-1]:.4f}")
    summary.append("")
    
    # Overall performance - VALIDATION
    summary.append("VALIDATION PERFORMANCE")
    summary.append("-" * 90)
    eval_initial = df_rounds.iloc[0]['eval_agg']
    eval_final = df_rounds.iloc[-1]['eval_agg']
    eval_best = df_rounds['eval_agg'].max()
    eval_best_round = df_rounds['eval_agg'].idxmax()
    
    summary.append(f"  Initial Validation Score:.................. {eval_initial:.4f}")
    summary.append(f"  Final Validation Score:.................... {eval_final:.4f}")
    summary.append(f"  Best Validation Score:..................... {eval_best:.4f} (Round {eval_best_round})")
    summary.append(f"  Validation Improvement:.................... {eval_final - eval_initial:.4f} ({(eval_final-eval_initial)/eval_initial*100:+.2f}%)")
    summary.append("")
    summary.append(f"  Best Validation mAP@0.5:................... {df_rounds['eval_mAP50'].max():.4f} (Round {df_rounds['eval_mAP50'].idxmax()})")
    summary.append(f"  Best Validation mAP:....................... {df_rounds['eval_mAP'].max():.4f} (Round {df_rounds['eval_mAP'].idxmax()})")
    summary.append(f"  Final Validation Loss:..................... {df_rounds['eval_loss'].iloc[-1]:.4f}")
    summary.append("")
    
    # Generalization analysis
    summary.append("GENERALIZATION ANALYSIS")
    summary.append("-" * 90)
    final_gap = df_rounds['train_agg'].iloc[-1] - df_rounds['eval_agg'].iloc[-1]
    avg_gap = (df_rounds['train_agg'] - df_rounds['eval_agg']).mean()
    max_gap = (df_rounds['train_agg'] - df_rounds['eval_agg']).max()
    
    summary.append(f"  Final Generalization Gap:.................. {final_gap:.4f}")
    summary.append(f"  Average Generalization Gap:................ {avg_gap:.4f}")
    summary.append(f"  Maximum Generalization Gap:................ {max_gap:.4f}")
    
    if avg_gap > 0.05:
        summary.append(f"  Status:.................................... OVERFITTING DETECTED")
    elif avg_gap > 0.02:
        summary.append(f"  Status:.................................... SLIGHT OVERFITTING")
    else:
        summary.append(f"  Status:.................................... GOOD GENERALIZATION")
    summary.append("")
    
    # Time statistics
    summary.append("TIME STATISTICS")
    summary.append("-" * 90)
    total_time = df_rounds['duration'].sum()
    avg_round = df_rounds['duration'].mean()
    
    summary.append(f"  Total Training Time:....................... {total_time/60:.2f} min ({total_time/3600:.2f} hours)")
    summary.append(f"  Average Round Duration:.................... {avg_round/60:.2f} min")
    summary.append(f"  Shortest Round:............................ {df_rounds['duration'].min()/60:.2f} min (Round {df_rounds['duration'].idxmin()})")
    summary.append(f"  Longest Round:............................. {df_rounds['duration'].max()/60:.2f} min (Round {df_rounds['duration'].idxmax()})")
    summary.append("")
    
    avg_client_train = df_clients['train_time'].mean()
    avg_client_eval = df_clients['eval_time'].mean()
    
    summary.append(f"  Avg Client Training Time:.................. {avg_client_train/60:.2f} min")
    summary.append(f"  Avg Client Eval Time:...................... {avg_client_eval:.2f} sec")
    summary.append(f"  Total Eval Time:........................... {df_rounds['eval_time'].sum():.2f} sec")
    summary.append("")
    
    # Communication
    summary.append("COMMUNICATION STATISTICS")
    summary.append("-" * 90)
    total_data = df_rounds['data_mb'].sum()
    avg_data = df_rounds['data_mb'].mean()
    
    summary.append(f"  Total Data Transferred:.................... {total_data:.2f} MB ({total_data/1024:.3f} GB)")
    summary.append(f"  Average per Round:......................... {avg_data:.2f} MB")
    summary.append(f"  Data Transfer Rate:........................ {total_data/(total_time/60):.2f} MB/min")
    summary.append("")
    
    # Client analysis
    summary.append("CLIENT ANALYSIS")
    summary.append("-" * 90)
    
    final_round = df_clients[df_clients['round_id'] == df_clients['round_id'].max()]
    
    # Training
    best_train_client = final_round.loc[final_round['train_agg'].idxmax()]
    worst_train_client = final_round.loc[final_round['train_agg'].idxmin()]
    
    summary.append(f"  Best Training Client:...................... Client {int(best_train_client['client_id'])} ({best_train_client['train_agg']:.4f})")
    summary.append(f"  Worst Training Client:..................... Client {int(worst_train_client['client_id'])} ({worst_train_client['train_agg']:.4f})")
    summary.append(f"  Training Performance Gap:.................. {best_train_client['train_agg'] - worst_train_client['train_agg']:.4f}")
    summary.append("")
    
    # Validation
    best_eval_client = final_round.loc[final_round['eval_agg'].idxmax()]
    worst_eval_client = final_round.loc[final_round['eval_agg'].idxmin()]
    
    summary.append(f"  Best Validation Client:.................... Client {int(best_eval_client['client_id'])} ({best_eval_client['eval_agg']:.4f})")
    summary.append(f"  Worst Validation Client:................... Client {int(worst_eval_client['client_id'])} ({worst_eval_client['eval_agg']:.4f})")
    summary.append(f"  Validation Performance Gap:................ {best_eval_client['eval_agg'] - worst_eval_client['eval_agg']:.4f}")
    summary.append("")
    
    # Key insights
    summary.append("KEY INSIGHTS")
    summary.append("-" * 90)
    
    # Training improvement
    if train_final > train_initial:
        summary.append(f"  [+] Training performance improved by {(train_final-train_initial)/train_initial*100:.1f}%")
    else:
        summary.append(f"  [-] Training performance decreased by {abs(train_final-train_initial)/train_initial*100:.1f}%")
    
    # Validation improvement
    if eval_final > eval_initial:
        summary.append(f"  [+] Validation performance improved by {(eval_final-eval_initial)/eval_initial*100:.1f}%")
    else:
        summary.append(f"  [-] Validation performance decreased by {abs(eval_final-eval_initial)/eval_initial*100:.1f}%")
    
    # Generalization
    if avg_gap < 0.02:
        summary.append(f"  [+] Excellent generalization (gap < 0.02)")
    elif avg_gap < 0.05:
        summary.append(f"  [!] Acceptable generalization (gap < 0.05)")
    else:
        summary.append(f"  [-] Poor generalization - overfitting detected (gap > 0.05)")
    
    # Convergence
    train_var_initial = df_clients[df_clients['round_id']==0]['train_agg'].std()
    train_var_final = final_round['train_agg'].std()
    
    if train_var_final < train_var_initial:
        summary.append(f"  [+] Clients are converging (variance reduced from {train_var_initial:.4f} to {train_var_final:.4f})")
    else:
        summary.append(f"  [!] Clients diverging (variance increased from {train_var_initial:.4f} to {train_var_final:.4f})")
    
    # Efficiency
    if eval_final > eval_initial:
        data_per_improvement = total_data / (eval_final - eval_initial)
        summary.append(f"  [*] Data efficiency: {data_per_improvement:.1f} MB per 0.01 improvement")
    
    summary.append("")
    summary.append("=" * 90)
    
    # Save report
    report_path = f"{output_dir}/00_SUMMARY_REPORT.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(summary))
    
    # Print to console
    print('\n'.join(summary))
    
    return report_path


# ==================== MAIN EXECUTION ====================

def main():
    """Main analysis pipeline"""
    print(f"\n{'='*90}")
    print(f"FEDERATED LEARNING EXPERIMENT ANALYZER")
    print(f"{'='*90}\n")
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    print(f"[OK] Output directory created: {output_dir}")
    
    # Load data
    print(f"\n[*] Loading data from: {INPUT_FILE}")
    data = load_data(INPUT_FILE)
    print(f"[OK] Loaded {len(data)} rounds")
    
    # Extract metrics
    print("\n[*] Extracting metrics...")
    df_rounds = extract_round_metrics(data)
    df_clients = extract_client_metrics(data)
    print(f"[OK] Extracted round-level metrics")
    print(f"[OK] Extracted client-level metrics ({len(df_clients)} records)")
    
    # Filter zero metrics clients
    print("\n[*] Filtering clients with zero metrics...")
    df_clients = filter_zero_metrics_clients(df_clients)
    
    # Generate visualizations
    print("\n[*] Generating visualizations...")
    
    plots = [
        ("Training vs Evaluation metrics", plot_train_vs_eval_metrics, (df_rounds,)),
        ("Generalization gap analysis", plot_generalization_gap, (df_rounds,)),
        ("Client performance heatmap", plot_client_performance_heatmap, (df_clients,)),
        ("Client train vs eval comparison", plot_client_train_vs_eval_comparison, (df_clients,)),
        ("Time analysis", plot_time_analysis, (df_clients, df_rounds)),
        ("Convergence analysis", plot_convergence_analysis, (df_rounds, df_clients)),
        ("Communication overhead", plot_communication_overhead, (df_rounds,)),
        ("Per-client analysis", plot_per_client_analysis, (df_clients,)),
    ]

    for desc, plot_func, args in plots:
        print(f"  • {desc}...")
        try:
            plot_func(*args, output_dir)
        except Exception as e:
            print(f"    ⚠️ Warning: {e}")
    
    # Generate summary
    print("\n[*] Generating summary report...")
    report_path = generate_summary_report(df_rounds, df_clients, output_dir)
    
    # Final summary
    num_plots = len(list(output_dir.glob('*.png')))
    print(f"\n{'='*90}")
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"{'='*90}")
    print(f"  • Generated {num_plots} visualization plots")
    print(f"  • Created summary report: {report_path}")
    print(f"  • All files saved to: {output_dir}/")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
