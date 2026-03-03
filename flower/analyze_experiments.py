#!/usr/bin/env python3
"""
Analyze federated learning experiments with partition and failure plan specifications.
Generates a comprehensive comparison table as .txt, .html, and .png outputs.
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


class ExperimentAnalyzer:
    def __init__(self, experiments_dir, partition_dir, failure_plans_dir, output_dir):
        self.experiments_dir = Path(experiments_dir)
        self.partition_dir = Path(partition_dir)
        self.failure_plans_dir = Path(failure_plans_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Cache for manifests and failure plans
        self.manifests = {}
        self.failure_plans = {}
        
    def get_failure_pct_from_exp_id(self, exp_id):
        """Extract failure percentage from experiment ID (last 2 digits)"""
        last_two = int(str(exp_id)[-2:])
        if last_two == 31:
            return 0  # No failure baseline
        elif last_two == 41:
            return 25
        elif last_two == 42:
            return 50
        elif last_two == 43:
            return 75
        else:
            return None
    
    def load_partition_manifest(self, dataset_id):
        """Load and cache partition manifest"""
        if dataset_id in self.manifests:
            return self.manifests[dataset_id]
        
        # Try different naming conventions (non-padded, 2-digit, 3-digit padding)
        possible_names = [
            f"partition_manifest_dataset_{dataset_id}.json",
            f"partition_manifest_dataset_{dataset_id:02d}.json",  # Zero-padded to 2 digits
            f"partition_manifest_dataset_{dataset_id:03d}.json",  # Zero-padded to 3 digits
        ]
        
        manifest_file = None
        for name in possible_names:
            candidate = self.partition_dir / name
            if candidate.exists():
                manifest_file = candidate
                break
        
        if not manifest_file:
            return None
        
        with open(manifest_file) as f:
            data = json.load(f)
            self.manifests[dataset_id] = data
            return data
    
    def load_failure_plan(self, failure_pct):
        """Load and cache failure plan"""
        if failure_pct is None or failure_pct == 0:
            return None
        
        if failure_pct in self.failure_plans:
            return self.failure_plans[failure_pct]
        
        plan_file = self.failure_plans_dir / f"failure_plan_dim4_{failure_pct}.json"
        if not plan_file.exists():
            return None
        
        with open(plan_file) as f:
            data = json.load(f)
            self.failure_plans[failure_pct] = data
            return data
    
    def parse_summary_report(self, report_path):
        """Extract key metrics from summary report"""
        metrics = {}
        
        with open(report_path) as f:
            content = f.read()
        
        # Extract experiment ID
        match = re.search(r'Experiment ID\.+\s*(\d+)', content)
        if match:
            metrics['exp_id'] = int(match.group(1))
        
        # Extract dataset ID
        match = re.search(r'Dataset\.+\s*(\d+)', content)
        if match:
            metrics['dataset_id'] = int(match.group(1))
        
        # Extract server rounds
        match = re.search(r'Server Rounds\.+\s*(\d+)', content)
        if match:
            metrics['server_rounds'] = int(match.group(1))
        
        # Final Training mAP@0.5
        match = re.search(r'Best Training mAP@0\.5:\.+\s*([\d.]+)', content)
        if match:
            metrics['train_map05'] = float(match.group(1))
        
        # Final Training Loss
        match = re.search(r'Final Training Loss:\.+\s*([\d.]+)', content)
        if match:
            metrics['train_loss'] = float(match.group(1))
        
        # Best Validation mAP@0.5
        match = re.search(r'Best Validation mAP@0\.5:\.+\s*([\d.]+)', content)
        if match:
            metrics['val_map05'] = float(match.group(1))
        
        # Final Validation Loss
        match = re.search(r'Final Validation Loss:\.+\s*([\d.]+)', content)
        if match:
            metrics['val_loss'] = float(match.group(1))
        
        # Total Training Time (extract minutes)
        match = re.search(r'Total Training Time:\.+\s*([\d.]+)\s*min', content)
        if match:
            metrics['training_time_min'] = float(match.group(1))
        
        # Total Data Transferred
        match = re.search(r'Total Data Transferred:\.+\s*([\d.]+)\s*MB', content)
        if match:
            metrics['data_transferred_mb'] = float(match.group(1))
        
        return metrics
    
    def extract_partition_specs(self, dataset_id):
        """Extract dataset specs from partition manifest"""
        manifest = self.load_partition_manifest(dataset_id)
        if not manifest:
            return {}
        
        meta = manifest.get('metadata', {})
        specs = {
            'dataset_description': meta.get('dataset_description', 'N/A'),
            'num_clients': meta.get('num_clients', 'N/A'),
            'min_train_per_client': meta.get('min_train_per_client', 'N/A'),
            'max_train_per_client': meta.get('max_train_per_client', 'N/A'),
        }
        
        # Calculate total samples
        partitions = manifest.get('partitions', {})
        total_train = sum(p.get('n_train_target', 0) for p in partitions.values())
        total_val = sum(p.get('n_val_target', 0) for p in partitions.values())
        specs['total_train_samples'] = total_train
        specs['total_val_samples'] = total_val
        
        return specs
    
    def extract_failure_specs(self, failure_pct):
        """Extract failure plan specs"""
        if failure_pct == 0:
            return {
                'failure_pct': 0,
                'num_failing_clients': 0,
                'num_rounds_with_failures': 0,
                'max_failure_duration': 0
            }
        
        plan = self.load_failure_plan(failure_pct)
        if not plan:
            return {}
        
        meta = plan.get('meta', {})
        episodes = plan.get('episodes', {})
        
        # Calculate rounds with failures
        rounds_with_failures = set()
        max_duration = 0
        for episode_id, episode in episodes.items():
            absent_rounds = episode.get('absent_rounds', [])
            rounds_with_failures.update(absent_rounds)
            duration = episode.get('down_duration', 0)
            max_duration = max(max_duration, duration)
        
        specs = {
            'failure_pct': meta.get('failure_pct', 'N/A'),
            'num_failing_clients': meta.get('num_failing_clients', 'N/A'),
            'num_rounds_with_failures': len(rounds_with_failures),
            'max_failure_duration': max_duration
        }
        
        return specs
    
    def analyze_all_experiments(self):
        """Analyze all experiments and compile results"""
        results = []
        
        # Find all summary report directories
        analysis_dirs = sorted([d for d in self.experiments_dir.iterdir() 
                               if d.is_dir() and d.name.startswith('analysis_exp_')])
        
        for analysis_dir in analysis_dirs:
            report_path = analysis_dir / '00_SUMMARY_REPORT.txt'
            if not report_path.exists():
                continue
            
            # Parse summary report
            metrics = self.parse_summary_report(report_path)
            if not metrics or 'exp_id' not in metrics:
                continue
            
            exp_id = metrics['exp_id']
            
            # Skip baseline experiment (112131)
            if exp_id == 112131:
                continue
            
            dataset_id = metrics.get('dataset_id')
            
            # Get failure percentage from exp_id
            failure_pct = self.get_failure_pct_from_exp_id(exp_id)
            
            # Extract partition specs
            partition_specs = self.extract_partition_specs(dataset_id) if dataset_id is not None else {}
            
            # Extract failure specs
            failure_specs = self.extract_failure_specs(failure_pct)
            
            # Compile row
            row = {
                'Exp ID': exp_id,
                'Dataset': dataset_id,
                'Dataset Type': partition_specs.get('dataset_description', 'N/A'),
                'Num Clients': partition_specs.get('num_clients', 'N/A'),
                'Total Train Samples': partition_specs.get('total_train_samples', 'N/A'),
                'Samples per Client': f"{partition_specs.get('min_train_per_client', 'N/A')}-{partition_specs.get('max_train_per_client', 'N/A')}",
                'Failure %': failure_specs.get('failure_pct', 'N/A'),
                'Num Failing Clients': failure_specs.get('num_failing_clients', 'N/A'),
                'Rounds w/ Failures': failure_specs.get('num_rounds_with_failures', 'N/A'),
                'Max Failure Duration': failure_specs.get('max_failure_duration', 'N/A'),
                'Server Rounds': metrics.get('server_rounds', 'N/A'),
                'Train mAP@0.5': round(metrics.get('train_map05', 0), 4),
                'Train Loss': round(metrics.get('train_loss', 0), 4),
                'Val mAP@0.5': round(metrics.get('val_map05', 0), 4),
                'Val Loss': round(metrics.get('val_loss', 0), 4),
                'Training Time (min)': round(metrics.get('training_time_min', 0), 2),
                'Data Transferred (MB)': round(metrics.get('data_transferred_mb', 0), 2),
            }
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def save_text_report(self, df, output_file):
        """Save results as formatted text file"""
        with open(output_file, 'w') as f:
            f.write("=" * 220 + "\n")
            f.write("FEDERATED LEARNING EXPERIMENTS COMPREHENSIVE ANALYSIS\n")
            f.write("=" * 220 + "\n\n")
            
            # Reorganize columns for better readability
            organized_cols = [
                'Exp ID',
                'Dataset',
                'Dataset Type',
                'Clients',
                'Min/Max Train',
                'Total Train',
                'Failure %',
                'Fail Clients',
                'Fail Rounds',
                'Max Fail Dur',
                'Rounds',
                'Time(min)',
                'Data(MB)',
                'Train mAP@0.5',
                'Val mAP@0.5',
                'Train Loss',
                'Val Loss',
            ]
            
            # Create a reorganized dataframe
            display_df = pd.DataFrame()
            
            for col in organized_cols:
                if col == 'Exp ID':
                    display_df['Exp ID'] = df['Exp ID']
                elif col == 'Dataset':
                    display_df['Dataset'] = df['Dataset']
                elif col == 'Dataset Type':
                    # Shorten dataset type descriptions
                    display_df['Dataset Type'] = df['Dataset Type'].apply(
                        lambda x: 'IID' if 'IID' in str(x) and 'non-IID' not in str(x) 
                        else 'Mixed' if 'both' in str(x) or ',' in str(x)
                        else 'Non-IID' if 'non-IID' in str(x)
                        else 'N/A'
                    )
                elif col == 'Clients':
                    display_df['Clients'] = df['Num Clients']
                elif col == 'Min/Max Train':
                    display_df['Min/Max Train'] = df['Samples per Client']
                elif col == 'Total Train':
                    display_df['Total Train'] = df['Total Train Samples']
                elif col == 'Failure %':
                    display_df['Failure %'] = df['Failure %']
                elif col == 'Fail Clients':
                    display_df['Fail Clients'] = df['Num Failing Clients']
                elif col == 'Fail Rounds':
                    display_df['Fail Rounds'] = df['Rounds w/ Failures']
                elif col == 'Max Fail Dur':
                    display_df['Max Fail Dur'] = df['Max Failure Duration']
                elif col == 'Rounds':
                    display_df['Rounds'] = df['Server Rounds']
                elif col == 'Time(min)':
                    display_df['Time(min)'] = df['Training Time (min)'].apply(lambda x: f"{x:.0f}" if isinstance(x, (int, float)) else x)
                elif col == 'Data(MB)':
                    display_df['Data(MB)'] = df['Data Transferred (MB)'].apply(lambda x: f"{x:.0f}" if isinstance(x, (int, float)) else x)
                elif col == 'Train mAP@0.5':
                    display_df['Train mAP@0.5'] = df['Train mAP@0.5'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                elif col == 'Val mAP@0.5':
                    display_df['Val mAP@0.5'] = df['Val mAP@0.5'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                elif col == 'Train Loss':
                    display_df['Train Loss'] = df['Train Loss'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                elif col == 'Val Loss':
                    display_df['Val Loss'] = df['Val Loss'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
            
            # Write organized table
            f.write(display_df.to_string(index=False))
            f.write("\n\n")
            
            # Detailed breakdown per experiment
            f.write("=" * 220 + "\n")
            f.write("DETAILED EXPERIMENT BREAKDOWN\n")
            f.write("=" * 220 + "\n\n")
            
            for idx, row in df.iterrows():
                f.write(f"Experiment {idx + 1}: ID {int(row['Exp ID'])}\n")
                f.write("-" * 220 + "\n")
                
                f.write(f"  DATASET:           {row['Dataset']} | Type: {row['Dataset Type']}\n")
                f.write(f"                     {row['Num Clients']} Clients, {row['Total Train Samples']:,} total samples, {row['Samples per Client']} per client\n\n")
                
                f.write(f"  FAILURE CONFIG:    {row['Failure %']}% failure rate\n")
                f.write(f"                     {row['Num Failing Clients']} clients fail, {row['Rounds w/ Failures']} rounds affected, max {row['Max Failure Duration']} round(s) downtime\n\n")
                
                f.write(f"  TRAINING CONFIG:   {int(row['Server Rounds'])} rounds, {row['Training Time (min)']:.2f} min total, {row['Data Transferred (MB)']:.2f} MB transferred\n\n")
                
                f.write(f"  PERFORMANCE:       Train mAP@0.5: {row['Train mAP@0.5']:.4f} | Val mAP@0.5: {row['Val mAP@0.5']:.4f}\n")
                f.write(f"                     Train Loss: {row['Train Loss']:.4f} | Val Loss: {row['Val Loss']:.4f}\n\n")
            
            # Summary statistics
            f.write("=" * 220 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 220 + "\n\n")
            
            numeric_cols = ['Train mAP@0.5', 'Val mAP@0.5', 'Train Loss', 'Val Loss', 'Training Time (min)', 'Data Transferred (MB)']
            for col in numeric_cols:
                if col in df.columns:
                    f.write(f"\n{col}:\n")
                    f.write(f"  Mean: {df[col].mean():.4f}\n")
                    f.write(f"  Std:  {df[col].std():.4f}\n")
                    f.write(f"  Min:  {df[col].min():.4f}\n")
                    f.write(f"  Max:  {df[col].max():.4f}\n")
    
    def save_interactive_html(self, df, output_file):
        """Save results as interactive HTML table"""
        # HTML with styled table and better layout
        html_content = """
        <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/simple-datatables@latest"></script>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/simple-datatables@latest/style.css">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 40px 20px;
                }
                .container {
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                    overflow: hidden;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                }
                .header h1 {
                    font-size: 28px;
                    margin-bottom: 10px;
                }
                .header p {
                    font-size: 14px;
                    opacity: 0.9;
                }
                .content {
                    padding: 40px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 30px;
                }
                thead th {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 15px;
                    text-align: left;
                    font-weight: 600;
                    font-size: 13px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    border: none;
                    position: sticky;
                    top: 0;
                }
                tbody td {
                    padding: 14px 15px;
                    border-bottom: 1px solid #E0E0E0;
                    font-size: 13px;
                }
                tbody tr {
                    transition: background-color 0.3s ease;
                }
                tbody tr:hover {
                    background-color: #F5F5F5;
                }
                tbody tr:nth-child(even) {
                    background-color: #FAFAFA;
                }
                .metrics {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                .metric-card {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }
                .metric-card .label {
                    font-size: 12px;
                    opacity: 0.9;
                    margin-bottom: 8px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                .metric-card .value {
                    font-size: 24px;
                    font-weight: bold;
                }
                .footer {
                    background: #F5F5F5;
                    padding: 20px 40px;
                    text-align: center;
                    color: #666;
                    font-size: 12px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Federated Learning Experiments Analysis</h1>
                    <p>Comprehensive comparison of all experiments with dataset and failure specifications</p>
                </div>
                <div class="content">
                    <div class="metrics">
        """
        
        # Add summary metrics
        html_content += f"""
                        <div class="metric-card">
                            <div class="label">Total Experiments</div>
                            <div class="value">{len(df)}</div>
                        </div>
                        <div class="metric-card">
                            <div class="label">Avg Train mAP@0.5</div>
                            <div class="value">{df['Train mAP@0.5'].mean():.4f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="label">Avg Val mAP@0.5</div>
                            <div class="value">{df['Val mAP@0.5'].mean():.4f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="label">Avg Training Time</div>
                            <div class="value">{df['Training Time (min)'].mean():.0f} min</div>
                        </div>
                    </div>

                    <h2 style="margin-bottom: 20px; color: #333; font-size: 18px;">Detailed Results</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Exp ID</th>
                                <th>Dataset</th>
                                <th>Type</th>
                                <th>Clients</th>
                                <th>Total Samples</th>
                                <th>Samples/Client</th>
                                <th>Failure %</th>
                                <th>Fail Clients</th>
                                <th>Fail Rounds</th>
                                <th>Rounds</th>
                                <th>Train mAP@0.5</th>
                                <th>Val mAP@0.5</th>
                                <th>Train Loss</th>
                                <th>Val Loss</th>
                                <th>Time (min)</th>
                                <th>Data (MB)</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        # Add table rows
        for _, row in df.iterrows():
            dataset_type = row['Dataset Type']
            if isinstance(dataset_type, str):
                if 'IID' in dataset_type and 'non-IID' not in dataset_type:
                    type_short = 'IID'
                elif 'non-IID' in dataset_type and ',' not in dataset_type:
                    type_short = 'Non-IID'
                elif 'both' in dataset_type or ',' in dataset_type:
                    type_short = 'Mixed'
                else:
                    type_short = 'N/A'
            else:
                type_short = 'N/A'
            
            html_content += f"""
                            <tr>
                                <td><strong>{int(row['Exp ID'])}</strong></td>
                                <td>{row['Dataset']}</td>
                                <td>{type_short}</td>
                                <td>{row['Num Clients']}</td>
                                <td>{int(row['Total Train Samples']):,}</td>
                                <td>{row['Samples per Client']}</td>
                                <td>{row['Failure %']}</td>
                                <td>{row['Num Failing Clients']}</td>
                                <td>{row['Rounds w/ Failures']}</td>
                                <td>{int(row['Server Rounds'])}</td>
                                <td><strong>{row['Train mAP@0.5']:.4f}</strong></td>
                                <td><strong>{row['Val mAP@0.5']:.4f}</strong></td>
                                <td>{row['Train Loss']:.4f}</td>
                                <td>{row['Val Loss']:.4f}</td>
                                <td>{row['Training Time (min)']:.0f}</td>
                                <td>{row['Data Transferred (MB)']:.0f}</td>
                            </tr>
            """
        
        html_content += """
                        </tbody>
                    </table>
                </div>
                <div class="footer">
                    <p>Generated on """  + pd.Timestamp.now().strftime("%B %d, %Y at %H:%M:%S") + """</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def save_png_plot(self, df, output_file):
        """Save results as PNG table visualization"""
        # Select key columns for visualization with better organization
        display_cols = [
            'Exp ID', 'Dataset', 'Num Clients', 'Total Train Samples',
            'Failure %', 'Num Failing Clients', 'Rounds w/ Failures',
            'Server Rounds', 'Training Time (min)',
            'Train mAP@0.5', 'Val mAP@0.5', 'Train Loss', 'Val Loss',
            'Data Transferred (MB)'
        ]
        
        # Filter columns that exist
        display_cols = [c for c in display_cols if c in df.columns]
        display_df = df[display_cols].copy()
        
        # Rename for shorter display
        rename_map = {
            'Num Clients': 'Clients',
            'Total Train Samples': 'Total Samples',
            'Num Failing Clients': 'Fail Clients',
            'Rounds w/ Failures': 'Fail Rounds',
            'Server Rounds': 'Rounds',
            'Training Time (min)': 'Time(min)',
            'Data Transferred (MB)': 'Data(MB)'
        }
        display_df = display_df.rename(columns=rename_map)
        
        # Format numbers
        for col in display_df.columns:
            if col in ['Train mAP@0.5', 'Val mAP@0.5', 'Train Loss', 'Val Loss']:
                display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}' if isinstance(x, (int, float)) else x)
            elif col in ['Time(min)', 'Data(MB)']:
                display_df[col] = display_df[col].apply(lambda x: f'{x:.0f}' if isinstance(x, (int, float)) else x)
            elif col == 'Total Samples':
                display_df[col] = display_df[col].apply(lambda x: f'{int(x):,}' if isinstance(x, (int, float)) else x)
        
        # Create figure with larger size for readability
        fig, ax = plt.subplots(figsize=(22, len(display_df) * 0.6 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=display_df.values,
                        colLabels=display_df.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.06] * len(display_df.columns))
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)
        
        # Style header
        for i in range(len(display_df.columns)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)
        
        # Alternate row colors and add gridlines
        for i in range(1, len(display_df) + 1):
            for j in range(len(display_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E8F1F5')
                else:
                    table[(i, j)].set_facecolor('#FFFFFF')
                
                # Add borders
                table[(i, j)].set_edgecolor('#CCCCCC')
                table[(i, j)].set_linewidth(0.5)
        
        plt.title('Federated Learning Experiments: Comprehensive Results Table', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"PNG saved to {output_file}")
    
    def run(self):
        """Execute full analysis"""
        print("Analyzing experiments...")
        df = self.analyze_all_experiments()
        
        if df.empty:
            print("No experiments found!")
            return
        
        print(f"Found {len(df)} experiments")
        
        # Save outputs
        txt_output = self.output_dir / 'experiments_analysis.txt'
        html_output = self.output_dir / 'experiments_analysis.html'
        png_output = self.output_dir / 'experiments_analysis.png'
        
        self.save_text_report(df, txt_output)
        print(f"Text report saved to {txt_output}")
        
        self.save_interactive_html(df, html_output)
        print(f"HTML report saved to {html_output}")
        
        self.save_png_plot(df, png_output)
        
        print(f"\n✓ Analysis complete!")
        print(f"  - {txt_output}")
        print(f"  - {html_output}")
        print(f"  - {png_output}")


if __name__ == '__main__':
    analyzer = ExperimentAnalyzer(
        experiments_dir='./experiments_outputs',
        partition_dir='./partition_outputs',
        failure_plans_dir='./failure_plans',
        output_dir='./experiments_outputs'
    )
    analyzer.run()
