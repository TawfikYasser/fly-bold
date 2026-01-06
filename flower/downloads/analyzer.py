import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import re


# ==================== CONFIGURATION ====================
INPUT_FILE = "EXP_YOLOv5_s_detection_37_logs.json"

# Extract experiment number
match = re.search(r"_detection_(\d+)_", INPUT_FILE)
exp_id = match.group(1) if match else "unknown"
OUTPUT_DIR = f"analysis_exp_{exp_id}"

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
    """Extract round-level evaluation metrics"""
    rounds = []
    for round_data in data:
        round_dict = {
            'round_id': round_data['round_id'],
            'duration': round_data['round_duration'],
            'lr': round_data['lr'],
            # Evaluation metrics (the real data)
            'eval_mr': round_data['round_eval_acc']['mr'],
            'eval_mp': round_data['round_eval_acc']['mp'],
            'eval_mAP50': round_data['round_eval_acc']['mAP@0.5'],
            'eval_mAP': round_data['round_eval_acc']['mAP'],
            'eval_agg': round_data['round_eval_acc']['aggregated'],
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
                # Evaluation metrics
                'eval_mr': client['client_eval_acc']['mr'],
                'eval_mp': client['client_eval_acc']['mp'],
                'eval_mAP50': client['client_eval_acc']['mAP@0.5'],
                'eval_mAP': client['client_eval_acc']['mAP'],
                'eval_agg': client['client_eval_acc']['aggregated'],
            }
            client_data.append(client_info)
    return pd.DataFrame(client_data)


# ==================== PLOTTING FUNCTIONS ====================

def plot_evaluation_metrics_over_rounds(df_rounds, output_dir):
    """Plot all evaluation metrics over rounds"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Federated Validation Performance Over Rounds', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    metrics = [
        ('eval_mr', 'Mean Recall (mR)', '#2E86AB'),
        ('eval_mp', 'Mean Precision (mP)', '#A23B72'),
        ('eval_mAP50', 'mAP@0.5', '#F18F01'),
        ('eval_mAP', 'mAP@0.5:0.95', '#C73E1D'),
        ('eval_agg', 'Aggregated Score', '#6A994E'),
    ]
    
    for idx, (metric, title, color) in enumerate(metrics):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        # Plot line
        ax.plot(df_rounds['round_id'], df_rounds[metric], 
                marker='o', linewidth=3, markersize=10, color=color, alpha=0.8)
        
        # Fill area under curve
        ax.fill_between(df_rounds['round_id'], 0, df_rounds[metric], 
                        alpha=0.2, color=color)
        
        # Add value annotations
        for i, row_data in df_rounds.iterrows():
            value = row_data[metric]
            ax.annotate(f'{value:.3f}', 
                       (row_data['round_id'], value),
                       textcoords="offset points", xytext=(0, 8), 
                       ha='center', fontsize=9, fontweight='bold')
        
        # Styling
        ax.set_xlabel('Round', fontsize=11, fontweight='bold')
        ax.set_ylabel(title.split('(')[0].strip(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(df_rounds['round_id'])
        
        # Add improvement indicator
        initial = df_rounds[metric].iloc[0]
        final = df_rounds[metric].iloc[-1]
        improvement = ((final - initial) / initial * 100)
        ax.text(0.02, 0.98, f'Δ {improvement:+.1f}%', 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove extra subplot
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_validation_metrics_over_rounds.png", 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_client_performance_heatmap(df_clients, output_dir):
    """Heatmap showing each client's performance across rounds"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Client Performance Evolution (Validation)', 
                 fontsize=15, fontweight='bold')
    
    metrics_data = [
        ('eval_agg', 'Aggregated Score', axes[0]),
        ('eval_mAP50', 'mAP@0.5', axes[1])
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
                           fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Client ID', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_client_performance_heatmap.png", 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_client_performance_distribution(df_clients, output_dir):
    """Box plots showing performance distribution across clients per round"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Client Performance Distribution by Round', 
                 fontsize=15, fontweight='bold')
    
    metrics = [
        ('eval_agg', 'Aggregated Score'),
        ('eval_mAP50', 'mAP@0.5'),
        ('eval_mAP', 'mAP@0.5:0.95'),
        ('eval_mr', 'Mean Recall'),
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        rounds = sorted(df_clients['round_id'].unique())
        data_by_round = [df_clients[df_clients['round_id'] == r][metric].values 
                        for r in rounds]
        
        # Box plot
        bp = ax.boxplot(data_by_round, labels=[f'R{r}' for r in rounds], 
                       patch_artist=True, widths=0.6)
        
        # Color boxes
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(rounds)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add mean line
        means = [np.mean(data) for data in data_by_round]
        ax.plot(range(1, len(rounds)+1), means, 'r-', linewidth=2.5, 
               marker='D', markersize=8, label='Mean', zorder=10)
        
        # Add median values as text
        medians = [np.median(data) for data in data_by_round]
        for i, (mean, median) in enumerate(zip(means, medians)):
            ax.text(i+1, mean, f'{mean:.3f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_client_performance_distribution.png", 
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
    
    plt.savefig(f"{output_dir}/04_time_analysis.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"{output_dir}/05_communication_overhead.png", 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_client_consistency(df_clients, output_dir):
    """Analyze client consistency and fairness"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Client Consistency & Fairness Analysis', 
                 fontsize=15, fontweight='bold')
    
    clients = sorted(df_clients['client_id'].unique())
    
    # 1. Performance variance per client
    ax1 = axes[0, 0]
    client_stats = []
    for c in clients:
        client_data = df_clients[df_clients['client_id'] == c]['eval_agg']
        client_stats.append({
            'client': c,
            'mean': client_data.mean(),
            'std': client_data.std(),
            'min': client_data.min(),
            'max': client_data.max()
        })
    
    stats_df = pd.DataFrame(client_stats)
    
    ax1.bar(stats_df['client'], stats_df['std'], color='#C73E1D', alpha=0.7)
    ax1.set_xlabel('Client ID', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Standard Deviation', fontsize=12)
    ax1.set_title('Performance Consistency (Lower = More Stable)', 
                 fontsize=12, fontweight='bold')
    ax1.set_xticks(clients)
    ax1.set_xticklabels([f'C{c}' for c in clients])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Mean vs Variance scatter
    ax2 = axes[0, 1]
    scatter = ax2.scatter(stats_df['mean'], stats_df['std'], 
                         s=300, alpha=0.6, c=stats_df['client'], 
                         cmap='viridis', edgecolors='black', linewidth=2)
    
    for _, row in stats_df.iterrows():
        ax2.annotate(f'C{int(row["client"])}', 
                    (row['mean'], row['std']),
                    fontsize=10, ha='center', va='center', 
                    fontweight='bold', color='white')
    
    ax2.set_xlabel('Mean Performance', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Performance Variability (Std)', fontsize=12)
    ax2.set_title('Performance vs Consistency Trade-off', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance range per client
    ax3 = axes[1, 0]
    x_pos = np.arange(len(clients))
    
    for i, c in enumerate(clients):
        client_data = df_clients[df_clients['client_id'] == c]['eval_agg']
        ax3.plot([i, i], [client_data.min(), client_data.max()], 
                'o-', linewidth=3, markersize=8, alpha=0.7)
        ax3.plot(i, client_data.mean(), 'r*', markersize=15, zorder=10)
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'C{c}' for c in clients])
    ax3.set_xlabel('Client ID', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Performance Range', fontsize=12)
    ax3.set_title('Performance Range (Min-Max, ★=Mean)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Client ranking evolution
    ax4 = axes[1, 1]
    rounds_list = sorted(df_clients['round_id'].unique())
    
    for c in clients:
        client_ranks = []
        for r in rounds_list:
            round_data = df_clients[df_clients['round_id'] == r]
            sorted_data = round_data.sort_values('eval_agg', ascending=False)
            rank = sorted_data[sorted_data['client_id'] == c].index[0]
            rank_pos = list(sorted_data.index).index(rank) + 1
            client_ranks.append(rank_pos)
        
        ax4.plot(rounds_list, client_ranks, marker='o', linewidth=2, 
                label=f'C{c}', markersize=6)
    
    ax4.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Rank (1=Best)', fontsize=12)
    ax4.set_title('Client Ranking Evolution', fontsize=12, fontweight='bold')
    ax4.invert_yaxis()  # Best rank at top
    ax4.set_xticks(rounds_list)
    ax4.set_yticks(range(1, len(clients)+1))
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/06_client_consistency.png", 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_convergence_analysis(df_rounds, df_clients, output_dir):
    """Convergence behavior analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Convergence Analysis', fontsize=15, fontweight='bold')
    
    # 1. Round-to-round improvement
    ax1 = axes[0, 0]
    metrics = ['eval_mAP50', 'eval_mAP', 'eval_agg']
    labels = ['mAP@0.5', 'mAP', 'Aggregated']
    colors = ['#F18F01', '#C73E1D', '#6A994E']
    
    for metric, label, color in zip(metrics, labels, colors):
        improvement = df_rounds[metric].diff()
        ax1.plot(df_rounds['round_id'][1:], improvement[1:], 
                marker='o', linewidth=2, label=label, color=color)
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Improvement from Previous Round', fontsize=12)
    ax1.set_title('Convergence Rate', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df_rounds['round_id'][1:])
    
    # 2. Cumulative improvement
    ax2 = axes[0, 1]
    for metric, label, color in zip(metrics, labels, colors):
        cumulative = df_rounds[metric] - df_rounds[metric].iloc[0]
        ax2.plot(df_rounds['round_id'], cumulative, 
                marker='s', linewidth=2.5, label=label, color=color, markersize=8)
    
    ax2.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Improvement', fontsize=12)
    ax2.set_title('Total Performance Gain', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(df_rounds['round_id'])
    
    # 3. Client convergence variance
    ax3 = axes[1, 0]
    variance_by_round = df_clients.groupby('round_id')['eval_agg'].std()
    
    ax3.plot(variance_by_round.index, variance_by_round.values, 
            marker='o', linewidth=3, markersize=10, color='#A23B72')
    ax3.fill_between(variance_by_round.index, 0, variance_by_round.values, 
                     alpha=0.3, color='#A23B72')
    
    ax3.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Client Performance Std Dev', fontsize=12)
    ax3.set_title('Client Convergence (Lower = More Aligned)', 
                 fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(variance_by_round.index)
    
    # 4. Learning curve with confidence interval
    ax4 = axes[1, 1]
    mean_perf = df_clients.groupby('round_id')['eval_agg'].mean()
    std_perf = df_clients.groupby('round_id')['eval_agg'].std()
    
    ax4.plot(mean_perf.index, mean_perf.values, 
            'o-', linewidth=3, markersize=10, color='#2E86AB', label='Mean')
    ax4.fill_between(mean_perf.index, 
                     mean_perf.values - std_perf.values,
                     mean_perf.values + std_perf.values,
                     alpha=0.3, color='#2E86AB', label='±1 Std Dev')
    
    ax4.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Performance (Aggregated)', fontsize=12)
    ax4.set_title('Learning Curve with Uncertainty', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(mean_perf.index)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_convergence_analysis.png", 
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
    
    # Overall performance
    summary.append("OVERALL PERFORMANCE (VALIDATION)")
    summary.append("-" * 90)
    initial = df_rounds.iloc[0]['eval_agg']
    final = df_rounds.iloc[-1]['eval_agg']
    best = df_rounds['eval_agg'].max()
    best_round = df_rounds['eval_agg'].idxmax()
    
    summary.append(f"  Initial Performance:........................ {initial:.4f}")
    summary.append(f"  Final Performance:.......................... {final:.4f}")
    summary.append(f"  Best Performance:........................... {best:.4f} (Round {best_round})")
    summary.append(f"  Total Improvement:.......................... {final - initial:.4f} ({(final-initial)/initial*100:+.2f}%)")
    summary.append("")
    
    summary.append(f"  Best mAP@0.5:............................... {df_rounds['eval_mAP50'].max():.4f} (Round {df_rounds['eval_mAP50'].idxmax()})")
    summary.append(f"  Best mAP@0.5:0.95:.......................... {df_rounds['eval_mAP'].max():.4f} (Round {df_rounds['eval_mAP'].idxmax()})")
    summary.append(f"  Best Recall:................................ {df_rounds['eval_mr'].max():.4f} (Round {df_rounds['eval_mr'].idxmax()})")
    summary.append(f"  Best Precision:............................. {df_rounds['eval_mp'].max():.4f} (Round {df_rounds['eval_mp'].idxmax()})")
    summary.append("")
    
    # Time statistics
    summary.append("TIME STATISTICS")
    summary.append("-" * 90)
    total_time = df_rounds['duration'].sum()
    avg_round = df_rounds['duration'].mean()
    
    summary.append(f"  Total Training Time:........................ {total_time/60:.2f} min ({total_time/3600:.2f} hours)")
    summary.append(f"  Average Round Duration:..................... {avg_round/60:.2f} min")
    summary.append(f"  Shortest Round:............................. {df_rounds['duration'].min()/60:.2f} min (Round {df_rounds['duration'].idxmin()})")
    summary.append(f"  Longest Round:.............................. {df_rounds['duration'].max()/60:.2f} min (Round {df_rounds['duration'].idxmax()})")
    summary.append("")
    
    avg_client_train = df_clients['train_time'].mean()
    avg_client_eval = df_clients['eval_time'].mean()
    
    summary.append(f"  Avg Client Training Time:................... {avg_client_train/60:.2f} min")
    summary.append(f"  Avg Client Eval Time:....................... {avg_client_eval:.2f} sec")
    summary.append(f"  Total Eval Time:............................ {df_rounds['eval_time'].sum():.2f} sec")
    summary.append("")
    
    # Communication
    summary.append("COMMUNICATION STATISTICS")
    summary.append("-" * 90)
    total_data = df_rounds['data_mb'].sum()
    avg_data = df_rounds['data_mb'].mean()
    
    summary.append(f"  Total Data Transferred:..................... {total_data:.2f} MB ({total_data/1024:.3f} GB)")
    summary.append(f"  Average per Round:.......................... {avg_data:.2f} MB")
    summary.append(f"  Data Transfer Rate:......................... {total_data/(total_time/60):.2f} MB/min")
    summary.append(f"  Data per Second:............................ {total_data*1024/(total_time):.2f} KB/sec")
    summary.append("")
    
    # Client analysis
    summary.append("CLIENT ANALYSIS")
    summary.append("-" * 90)
    
    # Final round stats
    final_round = df_clients[df_clients['round_id'] == df_clients['round_id'].max()]
    best_client_row = final_round.loc[final_round['eval_agg'].idxmax()]
    worst_client_row = final_round.loc[final_round['eval_agg'].idxmin()]
    
    summary.append(f"  Number of Clients:.......................... {len(final_round)}")
    summary.append(f"  Best Performing Client:..................... Client {int(best_client_row['client_id'])} ({best_client_row['eval_agg']:.4f})")
    summary.append(f"  Worst Performing Client:.................... Client {int(worst_client_row['client_id'])} ({worst_client_row['eval_agg']:.4f})")
    summary.append(f"  Performance Gap:............................ {best_client_row['eval_agg'] - worst_client_row['eval_agg']:.4f}")
    summary.append(f"  Mean Performance:........................... {final_round['eval_agg'].mean():.4f}")
    summary.append(f"  Std Dev:.................................... {final_round['eval_agg'].std():.4f}")
    summary.append("")
    
    # Most improved client
    clients = df_clients['client_id'].unique()
    improvements = []
    for c in clients:
        client_data = df_clients[df_clients['client_id'] == c]
        initial_perf = client_data.iloc[0]['eval_agg']
        final_perf = client_data.iloc[-1]['eval_agg']
        improvements.append((c, final_perf - initial_perf))
    
    improvements.sort(key=lambda x: x[1], reverse=True)
    best_improved = improvements[0]
    worst_improved = improvements[-1]
    
    summary.append(f"  Most Improved Client:....................... Client {int(best_improved[0])} ({best_improved[1]:+.4f})")
    summary.append(f"  Least Improved Client:...................... Client {int(worst_improved[0])} ({worst_improved[1]:+.4f})")
    summary.append("")
    
    # Convergence metrics
    summary.append("CONVERGENCE METRICS")
    summary.append("-" * 90)
    
    # Calculate convergence rate (average improvement per round)
    round_improvements = df_rounds['eval_agg'].diff().dropna()
    avg_improvement = round_improvements.mean()
    
    summary.append(f"  Average Round Improvement:.................. {avg_improvement:.4f}")
    summary.append(f"  Largest Single Improvement:................. {round_improvements.max():.4f} (Round {round_improvements.idxmax()})")
    summary.append(f"  Client Variance (Initial):.................. {df_clients[df_clients['round_id']==0]['eval_agg'].std():.4f}")
    summary.append(f"  Client Variance (Final):.................... {final_round['eval_agg'].std():.4f}")
    
    # Convergence indicator
    variance_trend = df_clients.groupby('round_id')['eval_agg'].std()
    converging = "YES" if variance_trend.iloc[-1] < variance_trend.iloc[0] else "NO"
    summary.append(f"  Clients Converging:......................... {converging}")
    summary.append("")
    
# Key insights
    summary.append("KEY INSIGHTS")
    summary.append("-" * 90)
    
    # Performance trajectory
    if final > initial:
        summary.append(f"  [+] Model performance improved by {(final-initial)/initial*100:.1f}%")
    else:
        summary.append(f"  [-] Model performance decreased by {abs(final-initial)/initial*100:.1f}%")
    
    # Convergence
    if converging == "YES":
        summary.append(f"  [+] Clients are converging (variance reduced)")
    else:
        summary.append(f"  [!] Clients are diverging (variance increased)")
    
    # Consistency
    if final_round['eval_agg'].std() < 0.05:
        summary.append(f"  [+] High client consistency (std < 0.05)")
    else:
        summary.append(f"  [!] Moderate client variance (std = {final_round['eval_agg'].std():.4f})")
    
    # Efficiency
    data_per_improvement = total_data / (final - initial) if final > initial else float('inf')
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
    
    # Generate visualizations
    print("\n[*] Generating visualizations...")
    
    plots = [
        ("Validation metrics over rounds", plot_evaluation_metrics_over_rounds),
        ("Client performance heatmap", plot_client_performance_heatmap),
        ("Client performance distribution", plot_client_performance_distribution),
        ("Time analysis", plot_time_analysis),
        ("Communication overhead", plot_communication_overhead),
        ("Client consistency", plot_client_consistency),
        ("Convergence analysis", plot_convergence_analysis),
    ]
    
    for desc, plot_func in plots:
        print(f"  - {desc}...")
        try:
            if plot_func == plot_time_analysis:
                plot_func(df_clients, df_rounds, output_dir)
            elif plot_func == plot_convergence_analysis:
                plot_func(df_rounds, df_clients, output_dir)
            else:
                plot_func(df_clients, output_dir) if 'client' in plot_func.__name__ else plot_func(df_rounds, output_dir)
        except Exception as e:
            print(f"    [!] Warning: {e}")
    
    # Generate summary
    print("\n[*] Generating summary report...")
    report_path = generate_summary_report(df_rounds, df_clients, output_dir)
    
    # Final summary
    num_plots = len(list(output_dir.glob('*.png')))
    print(f"\n{'='*90}")
    print(f"[SUCCESS] ANALYSIS COMPLETE!")
    print(f"{'='*90}")
    print(f"  - Generated {num_plots} visualization plots")
    print(f"  - Created summary report: {report_path}")
    print(f"  - All files saved to: {output_dir}/")
    print(f"{'='*90}\n")

if __name__ == "__main__":
    main()