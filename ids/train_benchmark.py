"""
IEC104 Benchmark Training Script
=================================
Converted from notebooks/iec104_benchmark.ipynb for HPC use.

Usage:
    python train_benchmark.py \
        --data-dir /scratch/j99tang/data/raw/iec104/iec104 \
        --output-dir /scratch/j99tang/results

Smoke test (fast, loads only N rows):
    python train_benchmark.py \
        --data-dir /scratch/j99tang/data/raw/iec104/iec104 \
        --output-dir /scratch/j99tang/test_output \
        --max-rows 5000
"""

# ── Must be FIRST before any other matplotlib import ──────────────────────
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no display needed on HPC
# ──────────────────────────────────────────────────────────────────────────

import argparse
import os
import warnings
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import sklearn
sklearn.set_config(enable_metadata_routing=False)

import mlflow
mlflow.autolog(disable=True)

from pycaret.classification import ClassificationExperiment
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score
)

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', '{:.4f}'.format)


# ── Command-line arguments ─────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='IEC104 benchmark training script')
    parser.add_argument(
        '--data-dir',
        default='/scratch/j99tang/data/raw/iec104/iec104',
        help='Path to the folder containing the IEC104 CSV files and headers_iec104.txt'
    )
    parser.add_argument(
        '--output-dir',
        default='/scratch/j99tang/results',
        help='Where to save model pkl and plot images'
    )
    parser.add_argument(
        '--max-rows',
        type=int,
        default=None,
        help='(Optional) Load only this many rows total — useful for a quick smoke test'
    )
    return parser.parse_args()


# ── Helper functions ───────────────────────────────────────────────────────

def read_headers(filename):
    """Read feature column names from the headers txt file."""
    with open(filename, 'r') as f:
        data = f.read()
    values = data.split(',\n')
    values[-1] = values[-1].rstrip('\n')
    return values


def load_data(iec104_dir, headers_file, max_rows=None):
    """Walk all CSV files under iec104_dir and concatenate them into one DataFrame."""
    cols = read_headers(headers_file)
    print(f'[1/5] Reading headers: {len(cols)} columns found')

    csv_files = []
    for root, dirs, files in os.walk(iec104_dir):
        for name in sorted(files):
            if name.endswith('.csv'):
                csv_files.append(os.path.join(root, name))

    print(f'[1/5] Found {len(csv_files)} CSV files:')
    for f in csv_files:
        print(f'      {os.path.relpath(f, iec104_dir)}')

    frames = []
    for path in csv_files:
        df_tmp = pd.read_csv(path, usecols=cols)
        frames.append(df_tmp)

    df = pd.concat(frames, ignore_index=True)

    if max_rows is not None:
        df = df.sample(n=min(max_rows, len(df)), random_state=123).reset_index(drop=True)
        print(f'[1/5] Smoke-test mode: sampled {len(df):,} rows')

    print(f'[1/5] Total rows loaded: {len(df):,}')
    return df, cols


def clean_data(df):
    """Drop NaN and infinite values (same as original pycaret_ids.py)."""
    df.dropna(axis=1, inplace=True)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
    print(f'[2/5] Rows after cleaning: {len(df):,} | Columns: {df.shape[1]}')
    return df


def print_class_distribution(df):
    class_counts = df['Label'].value_counts().sort_values(ascending=False)
    class_pct    = (class_counts / len(df) * 100).round(2)
    summary = pd.DataFrame({'Count': class_counts, 'Percentage (%)': class_pct})
    print('[2/5] Class distribution:')
    print(summary.to_string())
    ratio = class_counts.iloc[0] / class_counts.iloc[-1]
    print(f'      Imbalance ratio (largest/smallest): {ratio:.0f}:1')


def save_class_distribution_plot(df, output_dir):
    class_counts = df['Label'].value_counts().sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].bar(class_counts.index, class_counts.values,
                color=sns.color_palette('tab10', len(class_counts)))
    axes[0].set_title('Class Distribution (linear scale)')
    axes[0].set_ylabel('Sample count')
    axes[0].set_xticklabels(class_counts.index, rotation=35, ha='right')

    axes[1].bar(class_counts.index, class_counts.values,
                color=sns.color_palette('tab10', len(class_counts)))
    axes[1].set_yscale('log')
    axes[1].set_title('Class Distribution (log scale)')
    axes[1].set_ylabel('Sample count (log)')
    axes[1].set_xticklabels(class_counts.index, rotation=35, ha='right')
    axes[1].yaxis.set_major_formatter(mticker.ScalarFormatter())

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[2/5] Saved class distribution plot: {out_path}')


def train_pycaret(df):
    """Set up PyCaret experiment and run compare_models."""
    print('[3/5] Setting up PyCaret experiment ...')
    exp = ClassificationExperiment()
    exp.setup(
        data                 = df,
        target               = 'Label',
        session_id           = 123,
        index                = False,
        use_gpu              = False,
        train_size           = 0.7,
        preprocess           = True,
        numeric_imputation   = 'mean',
        normalize            = True,
        normalize_method     = 'zscore',
        fix_imbalance        = True,   # SMOTE — addresses class imbalance
        fold_strategy        = 'stratifiedkfold',
        fold                 = 5,
        fold_shuffle         = True,
        log_experiment       = False,
        log_plots            = False,
        profile              = False,
        memory               = True,
        max_encoding_ohe     = 0,
        verbose              = True
    )
    print('[3/5] PyCaret setup complete. Starting compare_models (this takes a while) ...')
    best_model = exp.compare_models(sort='MCC', verbose=True)
    print(f'[3/5] Best model: {type(best_model).__name__}')
    return exp, best_model


def evaluate_model(exp, best_model, output_dir):
    """Predict on the held-out test set and print metrics."""
    print('[4/5] Evaluating best model on test set ...')
    predictions = exp.predict_model(best_model)
    y_true = predictions['Label']
    y_pred = predictions['prediction_label']

    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, average='weighted')
    mcc   = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    print('=== Benchmark Model — Test Set Performance ===')
    print(f'Accuracy : {acc:.4f}')
    print(f'F1 (wtd) : {f1:.4f}')
    print(f'MCC      : {mcc:.4f}')
    print(f'Kappa    : {kappa:.4f}')
    print()
    print('Per-class report:')
    print(classification_report(y_true, y_pred, zero_division=0))

    # Paper baseline comparison
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 (weighted)', 'MCC', 'Kappa'],
        'Paper LDA (baseline)': [0.8566, 0.8546, 0.4266, 0.4264],
        'Our Benchmark': [acc, f1, mcc, kappa]
    })
    comparison['Delta'] = comparison['Our Benchmark'] - comparison['Paper LDA (baseline)']
    comparison = comparison.set_index('Metric')
    print('=== Comparison vs Paper Baseline ===')
    print(comparison.to_string())

    # Confusion matrix plot
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens',
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_title(f'Benchmark Model ({type(best_model).__name__}) — Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.xticks(rotation=40, ha='right')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix_benchmark.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[4/5] Saved confusion matrix: {cm_path}')

    return acc, f1, mcc, kappa


def save_model(exp, best_model, output_dir):
    """Save the trained PyCaret pipeline to a pkl file."""
    timestamp  = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = f'benchmark_iec104_{type(best_model).__name__.lower()}_{timestamp}'
    save_path  = os.path.join(output_dir, model_name)
    exp.save_model(best_model, model_name=save_path)
    print(f'[5/5] Model saved: {save_path}.pkl')


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    iec104_dir   = args.data_dir
    output_dir   = args.output_dir
    headers_file = os.path.join(iec104_dir, 'headers_iec104.txt')

    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print('IEC104 Benchmark Training')
    print(f'  Data dir   : {iec104_dir}')
    print(f'  Output dir : {output_dir}')
    if args.max_rows:
        print(f'  Max rows   : {args.max_rows} (SMOKE TEST MODE)')
    print('=' * 60)

    # 1. Load data
    df, cols = load_data(iec104_dir, headers_file, max_rows=args.max_rows)

    # 2. Clean and inspect
    df = clean_data(df)
    print_class_distribution(df)
    save_class_distribution_plot(df, output_dir)

    # 3. Train with PyCaret
    exp, best_model = train_pycaret(df)

    # 4. Evaluate
    evaluate_model(exp, best_model, output_dir)

    # 5. Save model
    save_model(exp, best_model, output_dir)

    print('=' * 60)
    print('All done.')
    print('=' * 60)


if __name__ == '__main__':
    main()
