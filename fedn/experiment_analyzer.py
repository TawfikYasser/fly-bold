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

    # 2. Fetch Validations
    validations_cursor = db['control.validations'].find()
    validations_data = list(validations_cursor)
    logger.info(f"Fetched {len(validations_data)} validations")

    # 3. Fetch Status (for client training time)
    status_cursor = db['control.status'].find({"status": "Model update completed."})
    status_data = list(status_cursor)
    logger.info(f"Fetched {len(status_data)} status completion logs")

    return rounds_data, validations_data, status_data

def export_to_json(df_rounds, df_clients, output_dir):
    """Reconstruct experiment log JSON from DataFrames"""
    try:
        export_data = []
        
        if df_rounds.empty:
            return None

        # Iterate through rounds to construct objects
        for _, row in df_rounds.iterrows():
            rid = int(row['round_id'])
            
            # Get client logs for this round
            round_clients = df_clients[df_clients['round_id'] == rid]
            client_logs = []
            for _, c_row in round_clients.iterrows():
                client_logs.append({
                    'name': str(c_row['client_id']),
                    'client_eval_num_examples': 500,
                    'client_train_num_examples': int(c_row.get('train_examples', 1000))
                })
            
            # Construct round object
            round_obj = {
                'round_id': rid,
                'round_duration': float(row.get('duration', 0)),
                'round_eval_acc': {
                    'aggregated': float(row.get('eval_agg', 0)),
                    'mAP@0.5': float(row.get('eval_mAP50', 0)),
                    'mAP': float(row.get('eval_mAP', 0)),
                    'mr': float(row.get('eval_mr', 0)),
                    'mp': float(row.get('eval_mp', 0))
                },
                'clients_logs': client_logs,
                'lr': 0.005
            }
            export_data.append(round_obj)
            
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"EXP_Reconstructed_{timestamp}_logs.json"
        filepath = output_dir / filename
        
        def default(o):
            if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
                return int(o)
            elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
                return float(o)
            elif isinstance(o, (np.ndarray,)):
                return o.tolist()
            return str(o)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=default)
            
        logger.info(f"Exported reconstructed logs to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to export JSON logs: {e}")
        print(f"CRITICAL EXPORT ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="FedN Analyzer")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    parser.add_argument("--out", default="analysis_plots", help="Output directory")
    parser.add_argument("--logs", help="Path to logs JSON file for detailed report")
    parser.add_argument("--dump-stdout", action="store_true", help="Print reconstructed logs to stdout")
    args = parser.parse_args()
    
    output_dir = Path(args.out)
    output_dir.mkdir(exist_ok=True)
    
    # Direct Log Parsing Mode (No DB)
    if args.logs:
        logger.info(f"Generating report from logs: {args.logs}")
        generate_detailed_report(args.logs, output_dir)
        return

    # Data Fetching
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

    # Stdout Dump Mode
    if args.dump_stdout:
        # Reconstruct data structure
        export_data = []
        for _, row in df_rounds.iterrows():
            rid = int(row['round_id'])
            round_clients = df_clients[df_clients['round_id'] == rid]
            client_logs = []
            for _, c_row in round_clients.iterrows():
                client_logs.append({
                    'name': str(c_row['client_id']),
                    'client_eval_num_examples': 500,
                    'client_train_num_examples': int(c_row.get('train_examples', 1000))
                })
            
            round_obj = {
                'round_id': rid,
                'round_duration': float(row.get('duration', 0)),
                'round_eval_acc': {
                    'aggregated': float(row.get('eval_agg', 0)),
                    'mAP@0.5': float(row.get('eval_mAP50', 0)),
                    'mAP': float(row.get('eval_mAP', 0)),
                    'mr': float(row.get('eval_mr', 0)),
                    'mp': float(row.get('eval_mp', 0))
                },
                'clients_logs': client_logs,
                'lr': 0.005
            }
            export_data.append(round_obj)
            
        def default(o):
            if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
                return int(o)
            elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
                return float(o)
            elif isinstance(o, (np.ndarray,)):
                return o.tolist()
            return str(o)
            
        # Print ONLY the JSON to stdout
        print(json.dumps(export_data, indent=2, default=default))
        return

    # Normal Analysis Mode
    # generate_summary_statistics(df_rounds, df_clients, output_dir)
    plot_metrics(df_rounds, df_clients, output_dir)
    
    # Export Reconstructed JSON and Generate Detailed Report
    json_path = export_to_json(df_rounds, df_clients, output_dir)
    if json_path:
        logger.info(f"Generating detailed report from reconstructed logs...")
        generate_detailed_report(json_path, output_dir)
    
    logger.info("Analysis complete.")

def process_data(rounds_data, validations_data, status_data=[]):
    """Process raw MongoDB data into structured DataFrames"""
    
    # --- Process Validations ---
    # Validation data map: model_id -> metrics
    # We need to link validation results to round_id
    
    # Create a map of model_id -> round_id
    model_to_round = {}
    
    # 1. Map from combiners
    for r in rounds_data:
        rid = r.get('round_id')
        if rid is None: continue
        
        combiners = r.get('combiners', [])
        if isinstance(combiners, list):
            for c in combiners:
                if isinstance(c, dict) and c.get('model_id'):
                    model_to_round[str(c.get('model_id'))] = rid
        if 'model_id' in r:
            model_to_round[str(r['model_id'])] = rid
            
    # 2. Heuristic temporal mapping for missing links
    # Identify unique validated models and sort by time
    validation_models = {} 
    for v in validations_data:
        mid = v.get('model_id') or v.get('modelId')
        ts = v.get('committed_at') or v.get('timestamp')
        if mid and ts:
            if mid not in validation_models or ts < validation_models[mid]:
                validation_models[mid] = ts
                
    sorted_val_models = sorted(validation_models.items(), key=lambda x: x[1])
    unique_val_model_ids = [m[0] for m in sorted_val_models]
    
    # Map strictly index -> index to rounds
    for idx, mid in enumerate(unique_val_model_ids):
        if idx < len(rounds_data):
            r = rounds_data[idx]
            rid = r.get('round_id')
            if str(mid) not in model_to_round:
               model_to_round[str(mid)] = rid

    # --- Extract Client Metrics ---
    client_records = []
    
    for v in validations_data:
        try:
            # Parse data
            data_str = v.get('data')
            if not data_str: continue
            
            data = json.loads(data_str) if isinstance(data_str, str) else data_str
            
            # Identify Client
            sender_name = v.get('sender', {}).get('name', 'unknown')
            client_id_match = re.search(r'(\d+)$', sender_name)
            client_id = client_id_match.group(1) if client_id_match else sender_name
            
            # Identify Round
            mid = v.get('model_id') or v.get('modelId')
            rid = v.get('round_id')
            
            round_id = v.get('round_id') # Use a new variable to avoid confusion with outer rid
            
            if round_id is None and mid and str(mid) in model_to_round:
                round_id = model_to_round[str(mid)]
            
            if round_id is None: continue
            
            if round_id is None:
                # Fallback: try to link via validation timestamp -> round timestamp
                # (omitted for brevity, relying on model_id map)
                pass

            # --- Extract Training Metrics (if available from previous validation-on-train run) ---
            # Our updated validate.py puts 'train_loss' etc in the main keys
            
            client_records.append({
                'round_id': getattr(round_id, 'item', lambda: round_id)() if hasattr(round_id, 'item') else round_id, # handle numpy scalar
                'client_id': client_id,
                'eval_mr': float(data.get('mr', data.get('eval_mr', np.nan))),
                'eval_mp': float(data.get('mp', data.get('eval_mp', np.nan))),
                'eval_mAP50': float(data.get('mAP@0.5', data.get('eval_mAP50', np.nan))),
                'eval_mAP': float(data.get('mAP', data.get('eval_mAP', np.nan))),
                'eval_agg': float(data.get('aggregated', data.get('eval_agg', data.get('mAP@0.5', np.nan)))), # Default agg to mAP@0.5 if not explicit
                'eval_examples': int(data.get('num_val_examples', 0)),
                'eval_loss': float(data.get('loss', data.get('eval_loss', np.nan))),
                
                # Training metrics (from validate.py validation-on-train)
                'train_loss': float(data.get('train_loss', np.nan)),
                'train_mAP': float(data.get('train_mAP', np.nan)),
                'train_mAP50': float(data.get('train_mAP@0.5', np.nan)),
                'train_mp': float(data.get('train_mp', np.nan)),
                'train_mr': float(data.get('train_mr', np.nan)),
                'train_agg': float(data.get('train_mAP@0.5', np.nan)), # Proxy agg
                'train_examples': int(data.get('num_train_examples', 0)),
                
                # Time Placeholders (will fill from status logs)
                'train_time': np.nan,
                'eval_time': np.nan
            })
            
        except Exception as e:
            continue
            
    # --- Process Status Logs for Training Metrics & Timing ---
    # Map: (round_id, client_id) -> {metrics...}
    client_status_map = {}
    
    for s in status_data:
        try:
            # Parse 'data' JSON string
            if not s.get('data'): continue
            s_data = json.loads(s['data'])
            
            # Identify Client
            sender = s_data.get('sender', {})
            c_id = sender.get('clientId') 
            if not c_id:
                 name = sender.get('name', '')
                 match = re.search(r'(\d+)$', name)
                 c_id = match.group(1) if match else name
            
            # Identify Round via 'meta' -> 'config' -> 'round_id'
            meta_str = s_data.get('meta')
            if meta_str:
                meta = json.loads(meta_str)
                
                # Config has round_id
                config_str = meta.get('config')
                if config_str:
                     config = json.loads(config_str)
                     r_id = config.get('round_id')
                     
                     if r_id is not None and c_id:
                         # Extraction
                         p_time = meta.get('processing_time')
                         training_meta = meta.get('training_metadata', {})
                         t_metrics = training_meta.get('metrics', {})
                         lr = training_meta.get('lr')
                         
                         key = (str(r_id), str(c_id))
                         if key not in client_status_map:
                             client_status_map[key] = {}
                         
                         if p_time:
                             client_status_map[key]['train_time'] = float(p_time)
                        
                         if lr is not None:
                             client_status_map[key]['lr'] = float(lr)
                             
                         # Extract metrics provided by updated train.py
                         for m_key, m_val in t_metrics.items():
                             # m_key like 'train_loss', 'train_mAP'
                             client_status_map[key][m_key] = float(m_val)

        except Exception as e:
            continue

    df_clients = pd.DataFrame(client_records)

    # Merge status info
    if not df_clients.empty:
        # Columns to potentially fill/overwrite
        cols_to_merge = ['train_time', 'lr', 'train_loss', 'train_mAP', 'train_mAP50', 'train_mp', 'train_mr']
        
        def merge_status_metrics(row):
            rid = str(row['round_id'])
            metrics = client_status_map.get((rid, str(row['client_id'])), {})
            
            updates = {}
            for col in cols_to_merge:
                # If available in status logs, use it (prioritize over validate.py workaround)
                if col in metrics and not pd.isna(metrics[col]):
                    updates[col] = metrics[col]
            return pd.Series(updates)

        # Apply updates
        status_updates = df_clients.apply(merge_status_metrics, axis=1)
        for col in status_updates.columns:
            # Fill NaNs or overwrite?
            # Prefer Status Metric if it exists.
            # If df_clients[col] is NaN, fill it.
            # If df_clients[col] has value (from validate.py), overwrite?
            # User implies native is better. Let's overwrite.
            if col in df_clients.columns:
                 df_clients[col].update(status_updates[col])
            else:
                 df_clients[col] = status_updates[col]

    # Drop duplicates if any
    if not df_clients.empty:
        df_clients.drop_duplicates(subset=['round_id', 'client_id'], keep='last', inplace=True)
    
    # --- Extract Round Metrics ---
    round_records = []
    
    for r in rounds_data:
        rid_raw = r.get('round_id')
        if rid_raw is None: continue
        try:
             rid = int(rid_raw)
        except:
             rid = rid_raw
        
        # Calculate Aggregated Eval Metrics for this round from clients
        # (Since rounds DB entry often lacks the aggregated result)
        round_clients = pd.DataFrame()
        if not df_clients.empty:
             # Ensure we compare same types (df_clients round_id is int)
             round_clients = df_clients[df_clients['round_id'] == rid]
        
        # Base round info
        rec = {
            'round_id': rid,
            'duration': float(r.get('round_duration', 0) if 'round_duration' in r else 0),
            'data_mb': float(r.get('data_transferred_mb', 0) if 'data_transferred_mb' in r else 0),
            'lr': np.nan, # Default
            
            # Train Metrics (Aggregate)
            'train_mr': np.nan,
            'train_mp': np.nan,
            'train_mAP50': np.nan,
            'train_mAP': np.nan,
            'train_agg': np.nan,
            'train_loss': np.nan,
            
            # Eval Metrics (Aggregate) - Default to NaN, fill if clients exist
            'eval_mr': np.nan,
            'eval_mp': np.nan,
            'eval_mAP50': np.nan,
            'eval_mAP': np.nan,
            'eval_agg': np.nan,
            'eval_loss': np.nan,
            'eval_time': np.nan
        }
        
        # Compute means from clients if available
        if not round_clients.empty:
            rec['eval_mr'] = round_clients['eval_mr'].mean()
            rec['eval_mp'] = round_clients['eval_mp'].mean()
            rec['eval_mAP50'] = round_clients['eval_mAP50'].mean()
            rec['eval_mAP'] = round_clients['eval_mAP'].mean()
            rec['eval_agg'] = round_clients['eval_agg'].mean()
            
            # Aggregate Training Metrics (from status/workaround)
            if 'train_loss' in round_clients: rec['train_loss'] = round_clients['train_loss'].mean()
            if 'train_mAP' in round_clients: rec['train_mAP'] = round_clients['train_mAP'].mean()
            if 'train_mAP50' in round_clients: rec['train_mAP50'] = round_clients['train_mAP50'].mean()
            if 'train_mp' in round_clients: rec['train_mp'] = round_clients['train_mp'].mean()
            if 'train_mr' in round_clients: rec['train_mr'] = round_clients['train_mr'].mean()
            if 'train_agg' in round_clients: rec['train_agg'] = round_clients['train_agg'].mean()
            if 'lr' in round_clients: rec['lr'] = round_clients['lr'].mean()
            
        round_records.append(rec)
        
    df_rounds = pd.DataFrame(round_records)
    
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
                'eval_time': random.uniform(1, 4), # Mock eval time
                'eval_examples': 500,
                'eval_loss': max(0.05, 1.0 - client_perf), # Mock eval loss
                
                # Training metrics (mocked)
                'train_mr': max(0, min(1, train_perf * 0.9)),
                'train_mp': max(0, min(1, train_perf * 1.1)),
                'train_mAP50': max(0, min(1, train_perf)),
                'train_mAP': max(0, min(1, train_perf * 0.6)),
                'train_agg': max(0, min(1, train_perf)),
                'train_loss': max(0.05, 1.0 - train_perf), # Mock train loss
                
                'train_examples': random.randint(1800, 3000), # Mock data size
                'train_time': random.uniform(5, 15) # Mock train time
            }
            records.append(rec)
            
    df_clients = pd.DataFrame(records)
    
    # Aggregates
    df_rounds = df_clients.groupby('round_id').mean(numeric_only=True).reset_index()
    # Mock durations and communication
    df_rounds['duration'] = [random.randint(100, 200) for _ in rounds]
    df_rounds['data_mb'] = [random.uniform(50, 150) for _ in rounds]
    
    return df_rounds, df_clients

# ==================== PLOTTING FUNCTIONS ====================

def plot_train_vs_eval_metrics(df_rounds, output_dir):
    """Plot training vs evaluation metrics side by side"""
    try:
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
            
            # Check if columns exist
            if f'train_{metric}' not in df_rounds.columns or f'eval_{metric}' not in df_rounds.columns:
                continue

            # Plot both train and eval
            ax.plot(df_rounds['round_id'], df_rounds[f'train_{metric}'], 
                    marker='o', linewidth=2.5, markersize=8, label='Training', 
                    color='#2E86AB', alpha=0.8)
            ax.plot(df_rounds['round_id'], df_rounds[f'eval_{metric}'], 
                    marker='s', linewidth=2.5, markersize=8, label='Validation', 
                    color='#A23B72', alpha=0.8)
            
            # Fill area between for visualization
            ax.fill_between(df_rounds['round_id'], 
                            df_rounds[f'train_{metric}'].fillna(0), 
                            df_rounds[f'eval_{metric}'].fillna(0),
                            alpha=0.2, color='gray')
            
            # Annotations on last point
            if not df_rounds.empty:
                last_train = df_rounds[f'train_{metric}'].iloc[-1]
                last_eval = df_rounds[f'eval_{metric}'].iloc[-1]
                last_round = df_rounds['round_id'].iloc[-1]
                
                if not np.isnan(last_train):
                    ax.annotate(f'{last_train:.3f}', 
                               (last_round, last_train),
                               textcoords="offset points", xytext=(5, 5), 
                               ha='left', fontsize=9, fontweight='bold', color='#2E86AB')
                if not np.isnan(last_eval):
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
        if 'train_loss' in df_rounds.columns and 'eval_loss' in df_rounds.columns:
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
    except Exception as e:
        logger.error(f"Error plotting comparison: {e}")


def plot_generalization_gap(df_rounds, output_dir):
    """Plot the generalization gap (train - eval) for all metrics"""
    try:
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
        
        valid_metrics_count = 0
        for idx, (metric, title) in enumerate(metrics):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            gap = None
            if metric == 'loss':
                 if 'train_loss' in df_rounds.columns and 'eval_loss' in df_rounds.columns:
                     gap = df_rounds['train_loss'] - df_rounds['eval_loss']
            else:
                 if f'train_{metric}' in df_rounds.columns and f'eval_{metric}' in df_rounds.columns:
                     gap = df_rounds[f'train_{metric}'] - df_rounds[f'eval_{metric}']

            if gap is None:
                continue
            
            valid_metrics_count += 1
            
            # Color based on whether gap is positive (overfitting) or negative
            colors = ['#C73E1D' if g > 0 else '#6A994E' for g in gap.fillna(0)]
            
            bars = ax.bar(df_rounds['round_id'], gap, color=colors, alpha=0.7, 
                         edgecolor='black', linewidth=1.5)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
            
            # Add value annotations
            for i, (r, g) in enumerate(zip(df_rounds['round_id'], gap)):
                if not np.isnan(g):
                    ax.text(r, g, f'{g:.3f}', 
                           ha='center', va='bottom' if g > 0 else 'top', 
                           fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Round', fontsize=11, fontweight='bold')
            ax.set_ylabel('Gap', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticks(df_rounds['round_id'])
            
            # Add interpretation text
            if not gap.empty:
                avg_gap = gap.mean()
                status = "Overfitting" if avg_gap > 0.02 else "Good Generalization"
                color = '#C73E1D' if avg_gap > 0.02 else '#6A994E'
                ax.text(0.02, 0.98, f'Avg: {avg_gap:.3f}\n{status}', 
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        if valid_metrics_count > 0:
            plt.tight_layout()
            plt.savefig(f"{output_dir}/02_generalization_gap.png", 
                        dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        logger.error(f"Error plotting generalization gap: {e}")


def plot_client_performance_heatmap(df_clients, output_dir):
    """Heatmap showing each client's performance across rounds"""
    try:
        if df_clients.empty: return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Client Performance Evolution', 
                     fontsize=15, fontweight='bold')
        
        metrics_data = [
            ('train_agg', 'Training Aggregated Score', axes[0, 0]),
            ('eval_agg', 'Validation Aggregated Score', axes[0, 1]),
            ('train_mAP50', 'Training mAP@0.5', axes[1, 0]),
            ('eval_mAP50', 'Validation mAP@0.5', axes[1, 1])
        ]
        
        valid_plots = 0
        for metric, title, ax in metrics_data:
            if metric not in df_clients.columns or 'round_id' not in df_clients.columns:
                continue
                
            pivot = df_clients.pivot(index='client_id', columns='round_id', values=metric)
            if pivot.empty: continue
            
            valid_plots += 1
            
            # Create heatmap
            min_val = pivot.values.min()
            max_val = pivot.values.max()
            if np.isnan(min_val): min_val = 0
            if np.isnan(max_val): max_val = 1
            
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', 
                          vmin=min_val*0.95, vmax=max_val*1.02)
            
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
                        ax.text(j, i, f'{value:.2f}',
                               ha="center", va="center", color=text_color, 
                               fontsize=8, fontweight='bold')
            
            ax.set_xlabel('Round', fontsize=12, fontweight='bold')
            ax.set_ylabel('Client ID', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        
        if valid_plots > 0:
            plt.tight_layout()
            plt.savefig(f"{output_dir}/03_client_performance_heatmap.png", 
                        dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        logger.error(f"Error plotting heatmap: {e}")


def plot_client_train_vs_eval_comparison(df_clients, output_dir):
    """Compare client Training vs Validation performance"""
    try:
        if df_clients.empty: return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Client Training vs Validation Performance', 
                     fontsize=15, fontweight='bold')
        
        # Last round data
        if 'round_id' not in df_clients.columns: return
        max_round = df_clients['round_id'].max()
        last_round = df_clients[df_clients['round_id'] == max_round]
        clients = sorted(last_round['client_id'].unique())
        
        if not clients: return
        
        # 1. Aggregated score comparison
        ax = axes[0, 0]
        x = np.arange(len(clients))
        width = 0.35
        
        train_scores = []
        eval_scores = []
        
        for c in clients:
            row = last_round[last_round['client_id'] == c]
            train_scores.append(row['train_agg'].values[0] if 'train_agg' in row else 0)
            eval_scores.append(row['eval_agg'].values[0] if 'eval_agg' in row else 0)
        
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
        train_map = []
        eval_map = []
        
        for c in clients:
            row = last_round[last_round['client_id'] == c]
            train_map.append(row['train_mAP50'].values[0] if 'train_mAP50' in row else 0)
            eval_map.append(row['eval_mAP50'].values[0] if 'eval_mAP50' in row else 0)
        
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
        all_vals = train_scores + eval_scores
        if all_vals:
            min_val = min(all_vals)
            max_val = max(all_vals)
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
        gaps = [t - e for t, e in zip(train_scores, eval_scores)]
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
    except Exception as e:
        logger.error(f"Error plotting train vs eval comparison: {e}")


def plot_time_analysis(df_clients, df_rounds, output_dir):
    """Comprehensive time analysis"""
    try:
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
        if 'train_time' in df_clients.columns:
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
        if 'eval_time' in df_clients.columns:
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
        if not df_clients.empty:
            last_round = df_clients[df_clients['round_id'] == df_clients['round_id'].max()]
            total_train = last_round['train_time'].sum() if 'train_time' in last_round else 0
            total_eval = last_round['eval_time'].sum() if 'eval_time' in last_round else 0
            
            if total_train + total_eval > 0:
                ax4.pie([total_train, total_eval], 
                       labels=['Training', 'Evaluation'],
                       autopct='%1.1f%%', startangle=90,
                       colors=['#A23B72', '#F18F01'])
                ax4.set_title(f'Time Distribution (Round {last_round["round_id"].iloc[0]})', 
                             fontsize=12, fontweight='bold')
        
        # 5. Client training time comparison
        ax5 = fig.add_subplot(gs[2, :])
        if 'train_time' in df_clients.columns:
            clients = sorted(df_clients['client_id'].unique())
            rounds_list = sorted(df_clients['round_id'].unique())
            
            x = np.arange(len(clients))
            width = 0.8 / max(1, len(rounds_list))
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
            ax5.legend(loc='upper right', fontsize=9, ncol=min(5, len(rounds_list)))
            ax5.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(f"{output_dir}/05_time_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting time analysis: {e}")


def plot_convergence_analysis(df_rounds, df_clients, output_dir):
    """Convergence behavior analysis"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Convergence Analysis', fontsize=15, fontweight='bold')
        
        # 1. Training improvement rate
        if 'train_agg' in df_rounds.columns and 'eval_agg' in df_rounds.columns:
            ax = axes[0, 0]
            train_improvement = df_rounds['train_agg'].diff()
            eval_improvement = df_rounds['eval_agg'].diff()
            
            if len(df_rounds) > 1:
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
            if len(df_rounds) > 1:
                ax.set_xticks(df_rounds['round_id'][1:])
        
        # 2. Cumulative improvement
        if 'train_agg' in df_rounds.columns and 'eval_agg' in df_rounds.columns:
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
        if 'train_loss' in df_rounds.columns and 'eval_loss' in df_rounds.columns:
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
        if 'train_agg' in df_clients.columns and 'eval_agg' in df_clients.columns:
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
        if 'train_agg' in df_clients.columns and 'eval_agg' in df_clients.columns:
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
        if 'train_agg' in df_rounds.columns and 'eval_agg' in df_rounds.columns:
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
    except Exception as e:
        logger.error(f"Error plotting convergence: {e}")


def plot_communication_overhead(df_rounds, output_dir):
    """Communication overhead analysis"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Communication Overhead', fontsize=15, fontweight='bold')
        
        # 1. Data transferred per round
        if 'data_mb' in df_rounds.columns:
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
    except Exception as e:
        logger.error(f"Error plotting communication overhead: {e}")


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
    # Using global CONFIG if available, or inferring
    # In FEDn script we might not have the separate CONFIG dict populated same way
    # We will infer what we can
    summary.append(f"  Total Clients:............................. {len(df_clients['client_id'].unique()) if not df_clients.empty else 0}")
    summary.append(f"  Server Rounds:............................. {len(df_rounds)}")
    summary.append("")
    
    if df_rounds.empty:
        print("No round data available for summary.")
        return

    # Overall performance - TRAINING
    summary.append("TRAINING PERFORMANCE")
    summary.append("-" * 90)
    
    if 'train_agg' in df_rounds.columns:
        train_initial = df_rounds.iloc[0]['train_agg']
        train_final = df_rounds.iloc[-1]['train_agg']
        train_best = df_rounds['train_agg'].max()
        train_best_round = df_rounds['train_agg'].idxmax()
        
        summary.append(f"  Initial Training Score:.................... {train_initial:.4f}")
        summary.append(f"  Final Training Score:...................... {train_final:.4f}")
        summary.append(f"  Best Training Score:....................... {train_best:.4f} (Round {train_best_round})")
        summary.append(f"  Training Improvement:...................... {train_final - train_initial:.4f} ({(train_final-train_initial)/train_initial*100:+.2f}%)")
        summary.append("")
        if 'train_mAP50' in df_rounds.columns:
            summary.append(f"  Best Training mAP@0.5:..................... {df_rounds['train_mAP50'].max():.4f} (Round {df_rounds['train_mAP50'].idxmax()})")
        if 'train_mAP' in df_rounds.columns:
            summary.append(f"  Best Training mAP:......................... {df_rounds['train_mAP'].max():.4f} (Round {df_rounds['train_mAP'].idxmax()})")
        if 'train_loss' in df_rounds.columns:
            summary.append(f"  Final Training Loss:....................... {df_rounds['train_loss'].iloc[-1]:.4f}")
        summary.append("")
    
    # Overall performance - VALIDATION
    summary.append("VALIDATION PERFORMANCE")
    summary.append("-" * 90)
    if 'eval_agg' in df_rounds.columns:
        eval_initial = df_rounds.iloc[0]['eval_agg']
        eval_final = df_rounds.iloc[-1]['eval_agg']
        eval_best = df_rounds['eval_agg'].max()
        eval_best_round = df_rounds['eval_agg'].idxmax()
        
        summary.append(f"  Initial Validation Score:.................. {eval_initial:.4f}")
        summary.append(f"  Final Validation Score:.................... {eval_final:.4f}")
        summary.append(f"  Best Validation Score:..................... {eval_best:.4f} (Round {eval_best_round})")
        summary.append(f"  Validation Improvement:.................... {eval_final - eval_initial:.4f} ({(eval_final-eval_initial)/eval_initial*100:+.2f}%)")
        summary.append("")
        if 'eval_mAP50' in df_rounds.columns:
            summary.append(f"  Best Validation mAP@0.5:................... {df_rounds['eval_mAP50'].max():.4f} (Round {df_rounds['eval_mAP50'].idxmax()})")
        if 'eval_mAP' in df_rounds.columns:
            summary.append(f"  Best Validation mAP:....................... {df_rounds['eval_mAP'].max():.4f} (Round {df_rounds['eval_mAP'].idxmax()})")
        if 'eval_loss' in df_rounds.columns:
            summary.append(f"  Final Validation Loss:..................... {df_rounds['eval_loss'].iloc[-1]:.4f}")
        summary.append("")
    
    # Generalization analysis
    if 'train_agg' in df_rounds.columns and 'eval_agg' in df_rounds.columns:
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
    if 'duration' in df_rounds.columns:
        total_time = df_rounds['duration'].sum()
        avg_round = df_rounds['duration'].mean()
        
        summary.append(f"  Total Training Time:....................... {total_time/60:.2f} min ({total_time/3600:.2f} hours)")
        summary.append(f"  Average Round Duration:.................... {avg_round/60:.2f} min")
        summary.append(f"  Shortest Round:............................ {df_rounds['duration'].min()/60:.2f} min (Round {df_rounds['duration'].idxmin()})")
        summary.append(f"  Longest Round:............................. {df_rounds['duration'].max()/60:.2f} min (Round {df_rounds['duration'].idxmax()})")
        summary.append("")
    
    if not df_clients.empty:
        if 'train_time' in df_clients.columns:
            avg_client_train = df_clients['train_time'].mean()
            summary.append(f"  Avg Client Training Time:.................. {avg_client_train/60:.2f} min")
        if 'eval_time' in df_clients.columns:
            avg_client_eval = df_clients['eval_time'].mean()
            summary.append(f"  Avg Client Eval Time:...................... {avg_client_eval:.2f} sec")
        if 'eval_time' in df_rounds.columns:
            summary.append(f"  Total Eval Time:........................... {df_rounds['eval_time'].sum():.2f} sec")
        summary.append("")
    
    # Communication
    if 'data_mb' in df_rounds.columns:
        summary.append("COMMUNICATION STATISTICS")
        summary.append("-" * 90)
        total_data = df_rounds['data_mb'].sum()
        avg_data = df_rounds['data_mb'].mean()
        
        summary.append(f"  Total Data Transferred:.................... {total_data:.2f} MB ({total_data/1024:.3f} GB)")
        summary.append(f"  Average per Round:......................... {avg_data:.2f} MB")
        if 'duration' in df_rounds.columns and df_rounds['duration'].sum() > 0:
            total_time_min = df_rounds['duration'].sum() / 60
            summary.append(f"  Data Transfer Rate:........................ {total_data/total_time_min:.2f} MB/min")
        summary.append("")
    
    # Client analysis
    if not df_clients.empty and 'train_agg' in df_clients.columns:
        summary.append("CLIENT ANALYSIS")
        summary.append("-" * 90)
        
        final_round_id = df_clients['round_id'].max()
        final_round = df_clients[df_clients['round_id'] == final_round_id]
        
        # Training
        if final_round['train_agg'].notna().any():
            best_train_idx = final_round['train_agg'].idxmax()
            worst_train_idx = final_round['train_agg'].idxmin()
            
            if pd.notna(best_train_idx) and pd.notna(worst_train_idx):
                best_train_client = final_round.loc[best_train_idx]
                worst_train_client = final_round.loc[worst_train_idx]
                
                summary.append(f"  Best Training Client:...................... Client {best_train_client['client_id']} ({best_train_client['train_agg']:.4f})")
                summary.append(f"  Worst Training Client:..................... Client {worst_train_client['client_id']} ({worst_train_client['train_agg']:.4f})")
                summary.append(f"  Training Performance Gap:.................. {best_train_client['train_agg'] - worst_train_client['train_agg']:.4f}")
            else:
                 summary.append("  Training Performance:...................... No data available")
        else:
             summary.append("  Training Performance:...................... No data available")
        summary.append("")
    
    if not df_clients.empty and 'eval_agg' in df_clients.columns:
        final_round_id = df_clients['round_id'].max()
        final_round = df_clients[df_clients['round_id'] == final_round_id]
        
        # Validation
        if final_round['eval_agg'].notna().any():
            best_eval_idx = final_round['eval_agg'].idxmax()
            worst_eval_idx = final_round['eval_agg'].idxmin()
            
            if pd.notna(best_eval_idx) and pd.notna(worst_eval_idx):
                best_eval_client = final_round.loc[best_eval_idx]
                worst_eval_client = final_round.loc[worst_eval_idx]
                
                summary.append(f"  Best Validation Client:.................... Client {best_eval_client['client_id']} ({best_eval_client['eval_agg']:.4f})")
                summary.append(f"  Worst Validation Client:................... Client {worst_eval_client['client_id']} ({worst_eval_client['eval_agg']:.4f})")
                summary.append(f"  Validation Performance Gap:................ {best_eval_client['eval_agg'] - worst_eval_client['eval_agg']:.4f}")
            else:
                 summary.append("  Validation Performance:.................... No data available")
        else:
            summary.append("  Validation Performance:.................... No data available")
        summary.append("")
    
    # Key insights
    summary.append("KEY INSIGHTS")
    summary.append("-" * 90)
    
    # Training improvement
    if 'train_agg' in df_rounds.columns:
        train_initial = df_rounds.iloc[0]['train_agg']
        train_final = df_rounds.iloc[-1]['train_agg']
        if train_final > train_initial:
            improvement = (train_final-train_initial)/train_initial*100 if train_initial != 0 else 0
            summary.append(f"  [+] Training performance improved by {improvement:.1f}%")
        else:
            decline = abs(train_final-train_initial)/train_initial*100 if train_initial != 0 else 0
            summary.append(f"  [-] Training performance decreased by {decline:.1f}%")
    
    # Validation improvement
    if 'eval_agg' in df_rounds.columns:
        eval_initial = df_rounds.iloc[0]['eval_agg']
        eval_final = df_rounds.iloc[-1]['eval_agg']
        if eval_final > eval_initial:
            improvement = (eval_final-eval_initial)/eval_initial*100 if eval_initial != 0 else 0
            summary.append(f"  [+] Validation performance improved by {improvement:.1f}%")
        else:
            decline = abs(eval_final-eval_initial)/eval_initial*100 if eval_initial != 0 else 0
            summary.append(f"  [-] Validation performance decreased by {decline:.1f}%")
    
    # Generalization
    if 'train_agg' in df_rounds.columns and 'eval_agg' in df_rounds.columns:
        avg_gap = (df_rounds['train_agg'] - df_rounds['eval_agg']).mean()
        if avg_gap < 0.02:
            summary.append(f"  [+] Excellent generalization (gap < 0.02)")
        elif avg_gap < 0.05:
            summary.append(f"  [!] Acceptable generalization (gap < 0.05)")
        else:
            summary.append(f"  [-] Poor generalization - overfitting detected (gap > 0.05)")
    
    # Convergence
    if 'train_agg' in df_clients.columns:
        train_var_initial = df_clients[df_clients['round_id']==df_clients['round_id'].min()]['train_agg'].std()
        final_round_id = df_clients['round_id'].max()
        train_var_final = df_clients[df_clients['round_id']==final_round_id]['train_agg'].std()
        
        if train_var_final < train_var_initial:
            summary.append(f"  [+] Clients are converging (variance reduced from {train_var_initial:.4f} to {train_var_final:.4f})")
        else:
            summary.append(f"  [!] Clients diverging (variance increased from {train_var_initial:.4f} to {train_var_final:.4f})")
    
    # Efficiency
    if 'eval_agg' in df_rounds.columns and 'data_mb' in df_rounds.columns:
        eval_initial = df_rounds.iloc[0]['eval_agg']
        eval_final = df_rounds.iloc[-1]['eval_agg']
        total_data = df_rounds['data_mb'].sum()
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

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description="FEDn Experiment Analyzer")
    parser.add_argument("--mock", action="store_true", help="Generate mock data instead of connecting to DB")
    parser.add_argument("--logs", type=str, help="Path to JSON logs file (bypasses DB connection)")
    parser.add_argument("--dump-stdout", action="store_true", help="Dump fetched data to stdout and exit")
    parser.add_argument("--out", type=str, default="analysis_plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rounds_data = []
    validations_data = []
    
    if args.mock:
        logger.info("Using MOCK data generator")
        df_rounds, df_clients = mock_data_generator()
        
    elif args.logs:
        logger.info(f"Loading logs from {args.logs}")
        try:
            with open(args.logs, 'r') as f:
                data = json.load(f)
            # Adapt schema if coming from merged logs or reconstructed
            # For now, let's assume we need to process it roughly like MongoDB or already formatted
            # Note: The current 'process_data' function expects raw mongo format (lists of dicts). 
            # If 'logs' is the reconstructed format (list of round objects), we might need adapting.
            # However, for simplicity, let's assume the user uses the 'merged_logs.json' which is structured.
            # We implemented 'process_data' to handle raw input.
            # If the input IS already df-like, we might need a parser.
            # CHECK: The reconstructed logs are a list of round objects.
            # We need a function to convert that list of dicts into df_rounds/df_clients matching Flower schema
            df_rounds, df_clients = process_logs_to_df(data)
            
        except Exception as e:
            logger.error(f"Failed to load logs: {e}")
            return
            
    else:
        # Default: Connect to Mongo
        db = get_mongo_connection()
        if db is None:
            return
        rounds_data, validations_data, status_data = fetch_data(db)
        
        if args.dump_stdout:
            # Reconstruct and dump
            # This logic was used to create the reconstructed_logs.json
            # We'll stick to the previous 'export_to_json' logic but print to stdout
            df_r, df_c = process_data(rounds_data, validations_data, status_data)
            reconstructed = []
            if not df_r.empty:
                for _, row in df_r.iterrows():
                    rid = int(row['round_id'])
                    round_clients = df_c[df_c['round_id'] == rid]
                    client_logs = []
                    for _, c_row in round_clients.iterrows():
                        client_logs.append({
                            'client_id': str(c_row['client_id']),
                             # Add back other client metrics if available
                        })
                    
                    round_obj = {
                        'round_id': rid,
                        'round_duration': float(row.get('duration', 0)),
                        'round_eval_acc': {
                            'aggregated': float(row.get('eval_agg', 0)),
                            'mAP@0.5': float(row.get('eval_mAP50', 0)),
                             # ... other metrics
                        },
                        'clients_logs': client_logs
                    }
                    if 'server_metrics' in row:
                        round_obj['round_data'] = row['server_metrics']
                    reconstructed.append(round_obj)
            
            print(json.dumps(reconstructed, indent=4))
            return
            
        df_rounds, df_clients = process_data(rounds_data, validations_data, status_data)
    
    print("Generating summary statistics...")
    generate_summary_report(df_rounds, df_clients, output_dir)

    print("  - Training vs Evaluation Metrics...")
    plot_train_vs_eval_metrics(df_rounds, output_dir)
    
    print("  - Generalization Gap Analysis...")
    plot_generalization_gap(df_rounds, output_dir)
    
    print("  - Client Performance Heatmaps...")
    plot_client_performance_heatmap(df_clients, output_dir)
    
    print("  - Client Training vs Evaluation Comparison...")
    plot_client_train_vs_eval_comparison(df_clients, output_dir)
    
    print("  - Time Analysis...")
    plot_time_analysis(df_clients, df_rounds, output_dir)
    
    print("  - Convergence Analysis...")
    plot_convergence_analysis(df_rounds, df_clients, output_dir)
    
    print("  - Communication Overhead...")
    plot_communication_overhead(df_rounds, output_dir)
    
    print(f"\n✓ Analysis complete! All plots saved to '{output_dir}/' directory")

def process_logs_to_df(data):
    """
    Convert the parsed JSON list (reconstructed logs) into df_rounds and df_clients
    matching the Flower schema.
    """
    rounds_list = []
    clients_list = []
    
    for r in data:
        # Round metrics
        rid = r.get('round_id')
        eval_acc = r.get('round_eval_acc', {})
        
        r_dict = {
            'round_id': rid,
            'duration': r.get('round_duration', 0),
            'data_mb': r.get('round_data_transferred_mb', 0),
            'train_agg': np.nan, # Default missing
            'train_mr': np.nan,
            'train_mp': np.nan,
            'train_mAP50': np.nan,
            'train_mAP': np.nan,
            'train_loss': np.nan,
            
            'eval_agg': eval_acc.get('aggregated'),
            'eval_mr': eval_acc.get('mr'),
            'eval_mp': eval_acc.get('mp'),
            'eval_mAP50': eval_acc.get('mAP@0.5'),
            'eval_mAP': eval_acc.get('mAP'),
            'eval_loss': np.nan,
        }
        rounds_list.append(r_dict)
        
        # Client metrics (if available in logs)
        if 'clients_logs' in r:
            for c in r['clients_logs']:
                # The reconstructed logs might have 'name' or 'client_id'
                cid = c.get('client_id') or c.get('name')
                rec = {
                    'round_id': rid,
                    'client_id': cid,
                    'train_time': c.get('client_train_time', 0),
                    'train_examples': c.get('client_train_num_examples', 0),
                    # Assume missing client-level metrics in basic logs
                    'train_agg': np.nan, 
                    'train_loss': np.nan,
                    'eval_loss': np.nan,
                }
                clients_list.append(rec)

    return pd.DataFrame(rounds_list), pd.DataFrame(clients_list)

if __name__ == "__main__":
    main()
