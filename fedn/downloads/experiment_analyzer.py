import os
import json
import argparse
import logging
import pymongo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration Defaults (can be overridden by env vars)
MONGO_HOST = os.environ.get('MONGO_HOST', 'mongo')
MONGO_PORT = int(os.environ.get('MONGO_PORT', 6534))
MONGO_USER = os.environ.get('MONGO_USER', 'fedn_admin')
MONGO_PASSWORD = os.environ.get('MONGO_PASSWORD', 'password')
NETWORK_ID = os.environ.get('NETWORK_ID', 'fedn-network')

def get_mongo_connection():
    """Connect to MongoDB"""
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

def fetch_data(db, limit=1000):
    """Fetch raw data from MongoDB collections"""
    logger.info("Fetching data from MongoDB...")
    
    # 1. Fetch Rounds
    rounds_cursor = db['control.rounds'].find().sort("round_id", 1)
    rounds_data = list(rounds_cursor)
    logger.info(f"Fetched {len(rounds_data)} rounds")

    # 2. Fetch Validations
    validations_cursor = db['control.validations'].find()
    validations_data = list(validations_cursor)
    logger.info(f"Fetched {len(validations_data)} validations")

    # 3. Fetch Metrics (for training data potentially)
    metrics_cursor = db['control.metrics'].find()
    metrics_data = list(metrics_cursor)
    logger.info(f"Fetched {len(metrics_data)} metrics")
    
    return rounds_data, validations_data, metrics_data

def process_data(rounds_data, validations_data, metrics_data):
    """Process raw MongoDB data into structured DataFrames"""
    
    # --- Process Rounds ---
    round_records = []
    
    # Map round IDs to duration/info
    round_info = {}
    for r in rounds_data:
        rid = r.get('round_id')
        # Calculate duration if available? 
        # FedN round objects might not have explicit duration stored simply, 
        # but let's check for 'combiners' data or similar. 
        # For now, we might mock duration or try to deduce it.
        # Let's just use what we have.
        round_info[str(rid)] = {
            'status': r.get('status'),
            # 'duration': ...  # Difficult to calculate without timestamps
        }

    # --- Process Validations (Evaluation Metrics) ---
    # Validation data in FedN typically looks like:
    # { 'model_id': ..., 'data': "{\"accuracy\": 0.9, ...}", 'sender': {'name': '...'}, ... }
    # Note: 'data' is often a stringified JSON.
    
    eval_records = []
    for v in validations_data:
        try:
            data = v.get('data')
            if isinstance(data, str):
                data = json.loads(data)
            
            # Extract common YOLO metrics if present
            # Adapting to what appeared in the Flower script: mr, mp, mAP50, mAP
            metrics = {
                'round_id': v.get('round_id', 'unknown'), # Needs link to round
                'client_id': v.get('sender', {}).get('name', 'unknown').replace('client', ''),
                'eval_mr': data.get('mr') or data.get('recall'),
                'eval_mp': data.get('mp') or data.get('precision'),
                'eval_mAP50': data.get('mAP@0.5') or data.get('mAP_0.5'),
                'eval_mAP': data.get('mAP') or data.get('mAP_0.5:0.95'),
            }
            # Add aggregated score if available or calculate
            metrics['eval_agg'] = metrics.get('eval_mAP50', 0) 
            
            # We need to associate validation with a round.
            # Usually validations refer to a model_id, and rounds produce a model_id.
            # Or the validation object itself might have round_id (check db schema).
            # The code above assumes 'round_id' might be in validation or we link via model_id.
            # If not in validation, map model_id -> round_id from round data.
            
            eval_records.append(metrics)
        except Exception:
            continue
            
    df_eval = pd.DataFrame(eval_records)
    
    # --- Process Metrics (Training Metrics) ---
    # Training metrics in FedN are often sent as "training_result" or similar via metadata
    # or separate metric entries.
    train_records = []
    for m in metrics_data:
        # Assuming metric structure
        try:
            # Example: {'name': 'training_loss', 'data': ..., 'round_id': ..., 'sender': ...}
            if m.get('name') == 'training_result': # Hypothetical name
                data = m.get('data')
                if isinstance(data, str):
                    data = json.loads(data)
                    
                metrics = {
                    'round_id': m.get('round_id'),
                    'client_id': m.get('sender', {}).get('name', 'unknown').replace('client', ''),
                    'train_mr': data.get('mr'),
                    'train_mp': data.get('mp'),
                    'train_mAP50': data.get('mAP@0.5'),
                    'train_mAP': data.get('mAP'),
                }
                metrics['train_agg'] = metrics.get('train_mAP50')
                train_records.append(metrics)
        except Exception:
            continue
            
    df_train = pd.DataFrame(train_records)

    # --- Merge and Aggregate ---
    # Create df_rounds (aggregated per round)
    # If we have client-level data, group by round_id
    
    # Combine train and eval
    # This is complex without exact schema, so we will create a best-effort Structure
    
    # MOCKING FOR ROBUSTNESS if empty (so visualization code doesn't crash)
    if df_eval.empty and df_train.empty:
        logger.warning("No metrics found. Generating empty/dummy frames for safety.")
        # create valid empty frames with columns
        cols = ['round_id', 'client_id', 'train_agg', 'eval_agg']
        df_clients = pd.DataFrame(columns=cols)
        df_rounds = pd.DataFrame(columns=['round_id', 'train_agg', 'eval_agg'])
        return df_rounds, df_clients

    # Merge client data
    # (Simplified merge logic)
    df_clients = pd.DataFrame()
    if not df_eval.empty:
        df_clients = df_eval.copy()
    
    if not df_train.empty:
        if df_clients.empty:
            df_clients = df_train.copy()
        else:
            # Merge on round_id and client_id
            df_clients = pd.merge(df_clients, df_train, on=['round_id', 'client_id'], how='outer')

    # Aggregated Round Data
    if not df_clients.empty:
        df_rounds = df_clients.groupby('round_id').mean(numeric_only=True).reset_index()
        # Add round info (duration etc)
        # df_rounds['duration'] = ...
    else:
         df_rounds = pd.DataFrame(columns=['round_id'])

    return df_rounds, df_clients

def mock_data():
    """Generate mock data for testing/offline usage"""
    logger.info("Generating MOCK data...")
    
    rounds = range(1, 11)
    clients = [str(i) for i in range(1, 11)]
    
    client_records = []
    
    import random
    
    for r in rounds:
        base_acc = 0.5 + (r * 0.04) # Improving accuracy
        for c in clients:
            # Random variation
            noise = random.uniform(-0.05, 0.05)
            c_acc = min(0.99, max(0.1, base_acc + noise))
            
            rec = {
                'round_id': r,
                'client_id': c,
                'train_time': random.uniform(5.0, 15.0),
                'train_examples': 1000,
                'train_mr': c_acc * 0.9,
                'train_mp': c_acc * 0.95,
                'train_mAP50': c_acc,
                'train_mAP': c_acc * 0.8,
                'train_agg': c_acc,
                
                'eval_time': random.uniform(2.0, 5.0),
                'eval_examples': 200,
                'eval_mr': (c_acc - 0.05) * 0.9,
                'eval_mp': (c_acc - 0.05) * 0.95,
                'eval_mAP50': c_acc - 0.05,
                'eval_mAP': (c_acc - 0.05) * 0.8,
                'eval_agg': c_acc - 0.05
            }
            client_records.append(rec)
            
    df_clients = pd.DataFrame(client_records)
    df_rounds = df_clients.groupby('round_id').mean(numeric_only=True).reset_index()
    # Add fake duration
    df_rounds['duration'] = [random.uniform(20, 30) for _ in rounds]
    
    return df_rounds, df_clients

# ==============================================================================
# VISUALIZATION FUNCTIONS (Ported/Adapted)
# ==============================================================================

def plot_round_metrics_comparison(df_rounds, output_dir):
    """Plot training vs evaluation metrics over rounds"""
    try:
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
            
            if f'train_{metric}' in df_rounds.columns:
                ax.plot(df_rounds['round_id'], df_rounds[f'train_{metric}'], 
                        marker='o', linewidth=2, label='Training', color='#2E86AB')
            if f'eval_{metric}' in df_rounds.columns:
                ax.plot(df_rounds['round_id'], df_rounds[f'eval_{metric}'], 
                        marker='s', linewidth=2, label='Validation', color='#A23B72')
            
            ax.set_xlabel('Round', fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        # Remove extra subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/01_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting comparison: {e}")

def plot_individual_metrics(df_rounds, output_dir):
    """Plot each metric separately"""
    metrics = [
        ('mr', 'Mean Recall (mR)', 'Recall'),
        ('mp', 'Mean Precision (mP)', 'Precision'),
        ('mAP50', 'mAP@0.5', 'mAP@0.5'),
        ('mAP', 'mAP@0.5:0.95', 'mAP'),
        ('agg', 'Aggregated Score', 'Score')
    ]
    
    for metric, title, ylabel in metrics:
        try:
            if f'train_{metric}' not in df_rounds.columns and f'eval_{metric}' not in df_rounds.columns:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            
            if f'train_{metric}' in df_rounds.columns:
                ax.plot(df_rounds['round_id'], df_rounds[f'train_{metric}'], 
                        marker='o', linewidth=2.5, markersize=8, label='Training', color='#2E86AB')
            if f'eval_{metric}' in df_rounds.columns:
                ax.plot(df_rounds['round_id'], df_rounds[f'eval_{metric}'], 
                        marker='s', linewidth=2.5, markersize=8, label='Validation', color='#A23B72')
            
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f'{title} Over Training Rounds', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/02_{metric}_trend.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting individual metric {metric}: {e}")

def plot_client_performance_heatmap(df_clients, output_dir):
    """Heatmap of client performance across rounds"""
    metrics = ['train_agg', 'eval_agg']
    titles = ['Client Training Performance (Aggregated)', 'Client Validation Performance (Aggregated)']
    
    for metric, title in zip(metrics, titles):
        try:
            if metric not in df_clients.columns:
                continue

            # Pivot data
            pivot_data = df_clients.pivot(index='client_id', columns='round_id', values=metric)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto')
            
            # Set ticks
            ax.set_xticks(np.arange(len(pivot_data.columns)))
            ax.set_yticks(np.arange(len(pivot_data.index)))
            ax.set_xticklabels(pivot_data.columns)
            ax.set_yticklabels([f'Client {i}' for i in pivot_data.index])
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Score', rotation=270, labelpad=20)
            
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel('Client ID', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/03_{metric}_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting heatmap {metric}: {e}")

def main():
    parser = argparse.ArgumentParser(description="FedN Experiment Analyzer")
    parser.add_argument("--mock", action="store_true", help="Use mock data instead of connecting to MongoDB")
    parser.add_argument("--out", type=str, default="analysis_plots", help="Output directory")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.mock:
        df_rounds, df_clients = mock_data()
    else:
        db = get_mongo_connection()
        if db is None:
            logger.error("Could not connect to DB. Creating mock data instead? No, aborting.")
            return
        
        rounds_raw, validations_raw, metrics_raw = fetch_data(db)
        df_rounds, df_clients = process_data(rounds_raw, validations_raw, metrics_raw)

    if df_rounds.empty:
        logger.error("No data available to plot.")
        return

    logger.info("Generating plots...")
    plot_round_metrics_comparison(df_rounds, output_dir)
    plot_individual_metrics(df_rounds, output_dir)
    plot_client_performance_heatmap(df_clients, output_dir)
    
    # Additional plots can be added here
    
    logger.info(f"Done! Plots saved to {output_dir}")

if __name__ == "__main__":
    main()
