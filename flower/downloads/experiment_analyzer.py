import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import re


# Configuration
INPUT_FILE = "EXP_YOLOv5_s_detection_14_logs.json"
# extract the experiment number (e.g. 14)
match = re.search(r"_detection_(\d+)_", INPUT_FILE)
exp_id = match.group(1) if match else "unknown"

OUTPUT_DIR = f"analysis_plots_{exp_id}"

# Experiment configuration
CONFIG = {
    "Train Images": 10000,
    "Val Images": 5000,
    "Server Rounds": 5,
    "Local Epochs": 2,
    "Batch Size": 24,
    "Learning Rate": 0.001,
    "YOLO Model": "s",
    "Image Size": 512,
    "Clients": 10,
    "Dirichlet Alpha": 0.7,
    "GPU Support": "NO"
}

def load_data(filename):
    """Load JSON data from file"""
    with open(filename, 'r') as f:
        return json.load(f)

def extract_round_metrics(data):
    """Extract aggregated metrics per round"""
    rounds = []
    for round_data in data:
        rounds.append({
            'round_id': round_data['round_id'],
            'duration': round_data['round_duration'],
            'train_examples': round_data['training_num_examples'],
            'lr': round_data['lr'],
            # Training metrics
            'train_mr': round_data['round_training_acc']['mr'],
            'train_mp': round_data['round_training_acc']['mp'],
            'train_mAP50': round_data['round_training_acc']['mAP@0.5'],
            'train_mAP': round_data['round_training_acc']['mAP'],
            'train_agg': round_data['round_training_acc']['aggregated'],
            # Eval metrics
            'eval_mr': round_data['round_eval_acc']['mr'],
            'eval_mp': round_data['round_eval_acc']['mp'],
            'eval_mAP50': round_data['round_eval_acc']['mAP@0.5'],
            'eval_mAP': round_data['round_eval_acc']['mAP'],
            'eval_agg': round_data['round_eval_acc']['aggregated']
        })
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
                # Training metrics
                'train_mr': client['client_train_acc']['mr'],
                'train_mp': client['client_train_acc']['mp'],
                'train_mAP50': client['client_train_acc']['mAP@0.5'],
                'train_mAP': client['client_train_acc']['mAP'],
                'train_agg': client['client_train_acc']['aggregated'],
            }
            # Add eval metrics if available
            if 'client_eval_acc' in client:
                client_info.update({
                    'eval_time': client['client_eval_time'],
                    'eval_examples': client['client_eval_num_example'],
                    'eval_mr': client['client_eval_acc']['mr'],
                    'eval_mp': client['client_eval_acc']['mp'],
                    'eval_mAP50': client['client_eval_acc']['mAP@0.5'],
                    'eval_mAP': client['client_eval_acc']['mAP'],
                    'eval_agg': client['client_eval_acc']['aggregated']
                })
            client_data.append(client_info)
    return pd.DataFrame(client_data)

def plot_round_metrics_comparison(df_rounds, output_dir):
    """Plot training vs evaluation metrics over rounds"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training vs Evaluation Metrics Over Rounds', fontsize=16, fontweight='bold')
    
    metrics = [
        ('mr', 'Mean Recall (mR)'),
        ('mp', 'Mean Precision (mP)'),
        ('mAP50', 'mAP@0.5'),
        ('mAP', 'mAP@0.5:0.95'),
        ('agg', 'Aggregated Score')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        ax.plot(df_rounds['round_id'], df_rounds[f'train_{metric}'], 
                marker='o', linewidth=2, label='Training', color='#2E86AB')
        ax.plot(df_rounds['round_id'], df_rounds[f'eval_{metric}'], 
                marker='s', linewidth=2, label='Validation', color='#A23B72')
        
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(df_rounds['round_id'])
    
    # Remove extra subplot
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_individual_metrics(df_rounds, output_dir):
    """Plot each metric separately for better visibility"""
    metrics = [
        ('mr', 'Mean Recall (mR)', 'Recall'),
        ('mp', 'Mean Precision (mP)', 'Precision'),
        ('mAP50', 'mAP@0.5', 'mAP@0.5'),
        ('mAP', 'mAP@0.5:0.95', 'mAP'),
        ('agg', 'Aggregated Score', 'Score')
    ]
    
    for metric, title, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(df_rounds['round_id'], df_rounds[f'train_{metric}'], 
                marker='o', linewidth=2.5, markersize=8, label='Training', color='#2E86AB')
        ax.plot(df_rounds['round_id'], df_rounds[f'eval_{metric}'], 
                marker='s', linewidth=2.5, markersize=8, label='Validation', color='#A23B72')
        
        # Add value annotations
        for i, row in df_rounds.iterrows():
            ax.annotate(f'{row[f"train_{metric}"]:.3f}', 
                       (row['round_id'], row[f'train_{metric}']),
                       textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            ax.annotate(f'{row[f"eval_{metric}"]:.3f}', 
                       (row['round_id'], row[f'eval_{metric}']),
                       textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{title} Over Training Rounds', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(df_rounds['round_id'])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/02_{metric}_trend.png", dpi=300, bbox_inches='tight')
        plt.close()

def plot_client_performance_heatmap(df_clients, output_dir):
    """Heatmap of client performance across rounds"""
    metrics = ['train_agg', 'eval_agg']
    titles = ['Client Training Performance (Aggregated)', 'Client Validation Performance (Aggregated)']
    
    for metric, title in zip(metrics, titles):
        # Create pivot table
        pivot_data = df_clients.pivot(index='client_id', columns='round_id', values=metric)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0.48, vmax=0.62)
        
        # Set ticks
        ax.set_xticks(np.arange(len(pivot_data.columns)))
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_xticklabels(pivot_data.columns)
        ax.set_yticklabels([f'Client {i}' for i in pivot_data.index])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Aggregated Score', rotation=270, labelpad=20)
        
        # Add value annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                value = pivot_data.values[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.3f}',
                                 ha="center", va="center", color="black", fontsize=9)
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Client ID', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/03_{metric}_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

def plot_client_metrics_per_round(df_clients, output_dir):
    """Box plots of client metrics for each round"""
    metrics = [
        ('train_agg', 'Training Aggregated Score'),
        ('eval_agg', 'Validation Aggregated Score'),
        ('train_mAP50', 'Training mAP@0.5'),
        ('eval_mAP50', 'Validation mAP@0.5')
    ]
    
    for metric, title in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data for box plot
        rounds = sorted(df_clients['round_id'].unique())
        data_by_round = [df_clients[df_clients['round_id'] == r][metric].dropna().values 
                        for r in rounds]
        
        bp = ax.boxplot(data_by_round, labels=[f'R{r}' for r in rounds], patch_artist=True)
        
        # Customize box plot colors
        for patch in bp['boxes']:
            patch.set_facecolor('#2E86AB')
            patch.set_alpha(0.7)
        
        # Add mean line
        means = [np.mean(data) for data in data_by_round]
        ax.plot(range(1, len(rounds)+1), means, 'r--', linewidth=2, marker='D', 
               markersize=8, label='Mean')
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{title} Distribution Across Clients', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/04_{metric}_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()

def plot_training_time_analysis(df_clients, df_rounds, output_dir):
    """Analyze training times"""
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
    axes[0].set_xticks(avg_train_time.index)
    
    # Plot 2: Total round duration
    axes[1].bar(df_rounds['round_id'], df_rounds['duration'], color='#A23B72', alpha=0.7)
    axes[1].set_xlabel('Round', fontsize=12)
    axes[1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1].set_title('Total Round Duration', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticks(df_rounds['round_id'])
    
    # Add time annotations
    for i, v in enumerate(df_rounds['duration']):
        axes[1].text(i, v, f'{v/3600:.2f}h', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_training_time_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_client_training_time_comparison(df_clients, output_dir):
    """Compare training times across clients"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    rounds = sorted(df_clients['round_id'].unique())
    clients = sorted(df_clients['client_id'].unique())
    
    x = np.arange(len(clients))
    width = 0.15
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    for i, round_id in enumerate(rounds):
        round_data = df_clients[df_clients['round_id'] == round_id]
        times = []
        for c in clients:
            row = round_data[round_data['client_id'] == c]
            if row.empty:
                times.append(0)  # or np.nan
            else:
                times.append(row['train_time'].iloc[0])
        ax.bar(x + i*width, times, width, label=f'Round {round_id}', color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Training Time per Client Across Rounds', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f'Client {c}' for c in clients])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/06_client_training_times.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_data_distribution(df_clients, output_dir):
    """Plot training data distribution across clients"""
    # Get data from last round
    last_round = df_clients['round_id'].max()
    last_round_data = df_clients[df_clients['round_id'] == last_round]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Training examples per client
    clients = sorted(last_round_data['client_id'].unique())
    examples = [last_round_data[last_round_data['client_id'] == c]['train_examples'].values[0] 
               for c in clients]
    
    axes[0].bar(clients, examples, color='#2E86AB', alpha=0.7)
    axes[0].set_xlabel('Client ID', fontsize=12)
    axes[0].set_ylabel('Number of Training Examples', fontsize=12)
    axes[0].set_title('Training Data Distribution (Non-IID)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_xticks(clients)
    
    # Add value annotations
    for i, v in enumerate(examples):
        axes[0].text(clients[i], v, str(v), ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Pie chart
    axes[1].pie(examples, labels=[f'C{c}' for c in clients], autopct='%1.1f%%',
               colors=plt.cm.Set3(np.linspace(0, 1, len(clients))))
    axes[1].set_title('Data Distribution Proportion', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_data_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_train_eval_gap(df_rounds, output_dir):
    """Plot the gap between training and evaluation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training-Validation Gap Analysis', fontsize=16, fontweight='bold')
    
    metrics = [
        ('mr', 'Mean Recall Gap'),
        ('mp', 'Mean Precision Gap'),
        ('mAP50', 'mAP@0.5 Gap'),
        ('mAP', 'mAP Gap')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
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
        ax.set_xticks(df_rounds['round_id'])
        
        # Add value annotations
        for i, (round_id, gap_val) in enumerate(zip(df_rounds['round_id'], gap)):
            ax.annotate(f'{gap_val:.3f}', (round_id, gap_val),
                       textcoords="offset points", xytext=(0,10 if gap_val > 0 else -15), 
                       ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/08_train_eval_gap.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_convergence_analysis(df_rounds, output_dir):
    """Analyze convergence behavior"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Improvement rate
    metrics = ['train_agg', 'eval_agg']
    labels = ['Training', 'Validation']
    colors = ['#2E86AB', '#A23B72']
    
    for metric, label, color in zip(metrics, labels, colors):
        improvement = df_rounds[metric].diff()
        axes[0].plot(df_rounds['round_id'][1:], improvement[1:], 
                    marker='o', linewidth=2, label=label, color=color)
    
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Round', fontsize=12)
    axes[0].set_ylabel('Improvement from Previous Round', fontsize=12)
    axes[0].set_title('Convergence Rate (Round-to-Round Improvement)', 
                     fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Cumulative improvement
    for metric, label, color in zip(metrics, labels, colors):
        cumulative = df_rounds[metric] - df_rounds[metric].iloc[0]
        axes[1].plot(df_rounds['round_id'], cumulative, 
                    marker='s', linewidth=2, label=label, color=color)
    
    axes[1].set_xlabel('Round', fontsize=12)
    axes[1].set_ylabel('Cumulative Improvement from Round 0', fontsize=12)
    axes[1].set_title('Cumulative Performance Gain', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(df_rounds['round_id'])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/09_convergence_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_client_consistency(df_clients, output_dir):
    """Analyze client consistency across rounds"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    clients = sorted(df_clients['client_id'].unique())
    
    # Plot 1: Training performance variance
    train_std = []
    train_mean = []
    for client_id in clients:
        client_data = df_clients[df_clients['client_id'] == client_id]['train_agg']
        train_std.append(client_data.std())
        train_mean.append(client_data.mean())
    
    axes[0].bar(clients, train_std, color='#2E86AB', alpha=0.7)
    axes[0].set_xlabel('Client ID', fontsize=12)
    axes[0].set_ylabel('Standard Deviation', fontsize=12)
    axes[0].set_title('Training Performance Consistency (Lower = More Consistent)', 
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_xticks(clients)
    
    # Plot 2: Mean vs Std scatter
    axes[1].scatter(train_mean, train_std, s=200, alpha=0.6, color='#A23B72')
    for i, client_id in enumerate(clients):
        axes[1].annotate(f'C{client_id}', (train_mean[i], train_std[i]),
                        fontsize=11, ha='center', va='center', color='white', 
                        fontweight='bold')
    
    axes[1].set_xlabel('Mean Training Performance', fontsize=12)
    axes[1].set_ylabel('Performance Variability (Std)', fontsize=12)
    axes[1].set_title('Client Performance: Mean vs Consistency', 
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/10_client_consistency.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_statistics(df_rounds, df_clients, output_dir):
    """Generate and save summary statistics"""
    summary = []
    
    summary.append("=" * 80)
    summary.append("FEDERATED LEARNING EXPERIMENT SUMMARY")
    summary.append("=" * 80)
    summary.append("")
    
    # Experiment configuration
    summary.append("EXPERIMENT CONFIGURATION:")
    summary.append("-" * 80)
    for key, value in CONFIG.items():
        summary.append(f"  {key:.<40} {value}")
    summary.append("")
    
    # Overall performance
    summary.append("OVERALL PERFORMANCE:")
    summary.append("-" * 80)
    initial_train = df_rounds.iloc[0]['train_agg']
    final_train = df_rounds.iloc[-1]['train_agg']
    initial_eval = df_rounds.iloc[0]['eval_agg']
    final_eval = df_rounds.iloc[-1]['eval_agg']
    
    summary.append(f"  Initial Training Score:................ {initial_train:.4f}")
    summary.append(f"  Final Training Score:.................. {final_train:.4f}")
    summary.append(f"  Training Improvement:.................. {final_train - initial_train:.4f} ({((final_train - initial_train)/initial_train*100):.2f}%)")
    summary.append("")
    summary.append(f"  Initial Validation Score:.............. {initial_eval:.4f}")
    summary.append(f"  Final Validation Score:................ {final_eval:.4f}")
    summary.append(f"  Validation Improvement:................ {final_eval - initial_eval:.4f} ({((final_eval - initial_eval)/initial_eval*100):.2f}%)")
    summary.append("")
    summary.append(f"  Best Validation mAP@0.5:............... {df_rounds['eval_mAP50'].max():.4f} (Round {df_rounds['eval_mAP50'].idxmax()})")
    summary.append(f"  Best Validation mAP:................... {df_rounds['eval_mAP'].max():.4f} (Round {df_rounds['eval_mAP'].idxmax()})")
    summary.append("")
    
    # Time statistics
    summary.append("TIME STATISTICS:")
    summary.append("-" * 80)
    total_time = df_rounds['duration'].sum()
    avg_round_time = df_rounds['duration'].mean()
    summary.append(f"  Total Experiment Time:................. {total_time/3600:.2f} hours")
    summary.append(f"  Average Round Duration:................ {avg_round_time/3600:.2f} hours")
    summary.append(f"  Average Client Training Time:.......... {df_clients['train_time'].mean():.2f} seconds")
    summary.append("")
    
    # Client statistics
    summary.append("CLIENT STATISTICS:")
    summary.append("-" * 80)
    client_final_round = df_clients[df_clients['round_id'] == df_rounds.iloc[-1]['round_id']]
    best_client = client_final_round.loc[client_final_round['train_agg'].idxmax()]
    worst_client = client_final_round.loc[client_final_round['train_agg'].idxmin()]
    
    summary.append(f"  Best Performing Client:................ Client {int(best_client['client_id'])} (Score: {best_client['train_agg']:.4f})")
    summary.append(f"  Worst Performing Client:............... Client {int(worst_client['client_id'])} (Score: {worst_client['train_agg']:.4f})")
    summary.append(f"  Performance Variance:.................. {client_final_round['train_agg'].std():.4f}")
    summary.append("")
    
    # Data distribution
    summary.append("DATA DISTRIBUTION:")
    summary.append("-" * 80)
    summary.append(f"  Min Examples per Client:............... {client_final_round['train_examples'].min():.0f}")
    summary.append(f"  Max Examples per Client:............... {client_final_round['train_examples'].max():.0f}")
    summary.append(f"  Average Examples per Client:........... {client_final_round['train_examples'].mean():.0f}")
    summary.append(f"  Data Imbalance Ratio:.................. {client_final_round['train_examples'].max() / client_final_round['train_examples'].min():.2f}x")
    summary.append("")
    
    summary.append("=" * 80)
    
    # Save to file
    with open(f"{output_dir}/00_summary_statistics.txt", 'w') as f:
        f.write('\n'.join(summary))
    
    # Print to console
    print('\n'.join(summary))

def main():
    """Main analysis function"""
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from {INPUT_FILE}...")
    data = load_data(INPUT_FILE)
    
    print("Extracting metrics...")
    df_rounds = extract_round_metrics(data)
    df_clients = extract_client_metrics(data)
    
    print("Generating summary statistics...")
    generate_summary_statistics(df_rounds, df_clients, output_dir)
    
    print("Creating visualizations...")
    print("  - Metrics comparison plot...")
    plot_round_metrics_comparison(df_rounds, output_dir)
    
    print("  - Individual metric trends...")
    plot_individual_metrics(df_rounds, output_dir)
    
    print("  - Client performance heatmaps...")
    plot_client_performance_heatmap(df_clients, output_dir)
    
    print("  - Client metrics distributions...")
    plot_client_metrics_per_round(df_clients, output_dir)
    
    print("  - Training time analysis...")
    plot_training_time_analysis(df_clients, df_rounds, output_dir)
    
    print("  - Client training time comparison...")
    plot_client_training_time_comparison(df_clients, output_dir)
    
    print("  - Data distribution plots...")
    plot_data_distribution(df_clients, output_dir)
    
    print("  - Train-eval gap analysis...")
    plot_train_eval_gap(df_rounds, output_dir)
    
    print("  - Convergence analysis...")
    plot_convergence_analysis(df_rounds, output_dir)
    
    print("  - Client consistency analysis...")
    plot_client_consistency(df_clients, output_dir)
    
    print(f"\n✓ Analysis complete! All plots saved to '{output_dir}/' directory")
    print(f"✓ Generated {len(list(output_dir.glob('*.png')))} plots and 1 summary file")

if __name__ == "__main__":
    main()