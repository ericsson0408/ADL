#!/usr/bin/env python3
"""
Plot training metrics for Cross-Encoder Reranker
Generates training loss and ranking evaluation curves
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_metrics_from_json(json_path):
    """Load training metrics from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def plot_reranker_curves(metrics, output_dir="./plots", model_name="reranker"):
    """
    Generate comprehensive training curves for reranker
    
    Args:
        metrics: Dict with keys ['steps', 'train_loss', 'mrr', 'ndcg', 'map', 'accuracy']
        output_dir: Directory to save plots
        model_name: Name for the plot title and filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Support both 'steps' and 'epochs' for backward compatibility
    x_values = metrics.get('steps', metrics.get('epochs', []))
    x_label = 'Training Steps' if 'steps' in metrics else 'Epoch'
    
    train_loss = metrics.get('train_loss', [])
    mrr = metrics.get('mrr', [])
    ndcg = metrics.get('ndcg', [])
    map_score = metrics.get('map', [])
    accuracy = metrics.get('accuracy', [])
    
    # Validate data
    if not x_values or not train_loss:
        print("Warning: Missing x_values or training loss data")
        return
    
    # ============================================================
    # Main Plot: Training Loss + Primary Ranking Metrics
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{model_name} Training Curves', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    ax1.plot(x_values[:len(train_loss)], train_loss, 'b-o', linewidth=2, markersize=8, label='Training Loss')
    ax1.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    
    # Add value annotations
    for i, (x, loss) in enumerate(zip(x_values[:len(train_loss)], train_loss)):
        ax1.annotate(f'{loss:.4f}', 
                    xy=(x, loss), 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # Plot 2: Ranking Metrics (MRR, NDCG, MAP)
    if mrr:
        ax2.plot(x_values[:len(mrr)], mrr, 's-', linewidth=2, markersize=8, label='MRR@10', color='#2ecc71')
    if ndcg:
        ax2.plot(x_values[:len(ndcg)], ndcg, '^-', linewidth=2, markersize=8, label='NDCG@10', color='#3498db')
    if map_score:
        ax2.plot(x_values[:len(map_score)], map_score, 'D-', linewidth=2, markersize=8, label='MAP@10', color='#e74c3c')
    if accuracy:
        ax2.plot(x_values[:len(accuracy)], accuracy, 'o-', linewidth=2, markersize=8, label='Accuracy@10', color='#9b59b6')
    
    ax2.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Ranking Evaluation Metrics', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11, loc='lower right')
    ax2.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    # Save main plot
    output_file = os.path.join(output_dir, f'{model_name}_training_curves.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Main plot saved: {output_file}")
    
    # Also save as PDF
    pdf_file = os.path.join(output_dir, f'{model_name}_training_curves.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"✓ PDF saved: {pdf_file}")
    
    plt.close()
    
    # ============================================================
    # Detailed Plot: 2x3 Grid for Individual Metrics
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{model_name} Detailed Training Metrics', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Plot 1: Training Loss (detailed)
    if train_loss:
        axes[0].plot(x_values[:len(train_loss)], train_loss, 'b-o', linewidth=2, markersize=6)
        axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        for i, (x, loss) in enumerate(zip(x_values[:len(train_loss)], train_loss)):
            axes[0].text(x, loss, f'{loss:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: MRR@10
    if mrr:
        axes[1].plot(x_values[:len(mrr)], mrr, 's-', linewidth=2, markersize=6, color='#2ecc71')
        axes[1].set_title('MRR@10 (Mean Reciprocal Rank)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel('MRR@10')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.0])
        for i, (x, score) in enumerate(zip(x_values[:len(mrr)], mrr)):
            axes[1].text(x, score, f'{score:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: NDCG@10
    if ndcg:
        axes[2].plot(x_values[:len(ndcg)], ndcg, '^-', linewidth=2, markersize=6, color='#3498db')
        axes[2].set_title('NDCG@10 (Normalized DCG)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel(x_label)
        axes[2].set_ylabel('NDCG@10')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1.0])
        for i, (x, score) in enumerate(zip(x_values[:len(ndcg)], ndcg)):
            axes[2].text(x, score, f'{score:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: MAP@10
    if map_score:
        axes[3].plot(x_values[:len(map_score)], map_score, 'D-', linewidth=2, markersize=6, color='#e74c3c')
        axes[3].set_title('MAP@10 (Mean Average Precision)', fontsize=12, fontweight='bold')
        axes[3].set_xlabel(x_label)
        axes[3].set_ylabel('MAP@10')
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim([0, 1.0])
        for i, (x, score) in enumerate(zip(x_values[:len(map_score)], map_score)):
            axes[3].text(x, score, f'{score:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 5: Accuracy@10
    if accuracy:
        axes[4].plot(x_values[:len(accuracy)], accuracy, 'o-', linewidth=2, markersize=6, color='#9b59b6')
        axes[4].set_title('Accuracy@10', fontsize=12, fontweight='bold')
        axes[4].set_xlabel(x_label)
        axes[4].set_ylabel('Accuracy@10')
        axes[4].grid(True, alpha=0.3)
        axes[4].set_ylim([0, 1.0])
        for i, (x, score) in enumerate(zip(x_values[:len(accuracy)], accuracy)):
            axes[4].text(x, score, f'{score:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Summary comparison
    if mrr and ndcg and map_score:
        x_indices = np.arange(len(x_values[:len(mrr)]))
        width = 0.25
        axes[5].bar(x_indices - width, mrr, width, label='MRR@10', color='#2ecc71', alpha=0.8)
        axes[5].bar(x_indices, ndcg, width, label='NDCG@10', color='#3498db', alpha=0.8)
        axes[5].bar(x_indices + width, map_score, width, label='MAP@10', color='#e74c3c', alpha=0.8)
        axes[5].set_title('Metrics Comparison', fontsize=12, fontweight='bold')
        axes[5].set_xlabel(x_label)
        axes[5].set_ylabel('Score')
        axes[5].set_xticks(x_indices)
        axes[5].set_xticklabels([str(x) for x in x_values[:len(mrr)]])
        axes[5].legend()
        axes[5].grid(True, alpha=0.3, axis='y')
        axes[5].set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    # Save detailed plot
    detailed_file = os.path.join(output_dir, f'{model_name}_metrics_detailed.png')
    plt.savefig(detailed_file, dpi=300, bbox_inches='tight')
    print(f"✓ Detailed plot saved: {detailed_file}")
    
    plt.close()

def print_summary_table(metrics):
    """Print a summary table of training metrics"""
    print("\n" + "="*70)
    print("RERANKER TRAINING METRICS SUMMARY")
    print("="*70)
    
    # Support both 'steps' and 'epochs' for backward compatibility
    x_values = metrics.get('steps', metrics.get('epochs', []))
    x_label = 'Step' if 'steps' in metrics else 'Epoch'
    
    train_loss = metrics.get('train_loss', [])
    mrr = metrics.get('mrr', [])
    ndcg = metrics.get('ndcg', [])
    map_score = metrics.get('map', [])
    accuracy = metrics.get('accuracy', [])
    
    # Header
    header = f"{x_label:<10} {'Train Loss':<15} {'MRR@10':<12} {'NDCG@10':<12} {'MAP@10':<12} {'Acc@10':<12}"
    print(header)
    print("-" * 70)
    
    # Data rows
    max_len = max(len(train_loss), len(mrr), len(ndcg), len(map_score), len(accuracy))
    for i in range(max_len):
        x_val = x_values[i] if i < len(x_values) else '-'
        loss = f"{train_loss[i]:.6f}" if i < len(train_loss) else '-'
        mrr_val = f"{mrr[i]:.4f}" if i < len(mrr) else '-'
        ndcg_val = f"{ndcg[i]:.4f}" if i < len(ndcg) else '-'
        map_val = f"{map_score[i]:.4f}" if i < len(map_score) else '-'
        acc_val = f"{accuracy[i]:.4f}" if i < len(accuracy) else '-'
        
        print(f"{x_val:<10} {loss:<15} {mrr_val:<12} {ndcg_val:<12} {map_val:<12} {acc_val:<12}")
    
    print("="*70)
    
    # Final metrics
    if train_loss:
        print(f"\n✓ Total Evaluation Points: {len(x_values)}")
        print(f"✓ Final Training Loss: {train_loss[-1]:.6f}")
        if len(train_loss) > 1:
            improvement = train_loss[0] - train_loss[-1]
            print(f"✓ Loss Improvement: {improvement:.6f} ({improvement/train_loss[0]*100:.2f}%)")
    
    if mrr:
        print(f"✓ Best MRR@10: {max(mrr):.4f} (at {x_label.lower()} {x_values[mrr.index(max(mrr))]})")
    if ndcg:
        print(f"✓ Best NDCG@10: {max(ndcg):.4f} (at {x_label.lower()} {x_values[ndcg.index(max(ndcg))]})")
    if map_score:
        print(f"✓ Best MAP@10: {max(map_score):.4f} (at {x_label.lower()} {x_values[map_score.index(max(map_score))]})")
    
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Plot reranker training metrics')
    parser.add_argument('--model_dir', type=str, default='reranker_2',
                        help='Directory containing training_metrics.json')
    parser.add_argument('--output_dir', type=str, default='./plots',
                        help='Directory to save plots')
    parser.add_argument('--model_name', type=str, default='Cross-Encoder Reranker',
                        help='Name for plot titles')
    args = parser.parse_args()
    
    # Load metrics
    metrics_file = os.path.join(args.model_dir, 'training_metrics.json')
    
    if not os.path.exists(metrics_file):
        print(f"Error: {metrics_file} not found!")
        print("\nGenerating synthetic data for demonstration...")
        
        # Generate synthetic data for demonstration (using steps instead of epochs)
        metrics = {
            "steps": [100, 200, 300, 400, 500],
            "train_loss": [0.4523, 0.3201, 0.2564, 0.2103, 0.1892],
            "mrr": [0.6234, 0.7012, 0.7456, 0.7623, 0.7701],
            "ndcg": [0.5987, 0.6834, 0.7234, 0.7456, 0.7523],
            "map": [0.5645, 0.6523, 0.6967, 0.7134, 0.7245],
            "accuracy": [0.5423, 0.6234, 0.6756, 0.6923, 0.7034]
        }
        print("Using synthetic metrics for demonstration")
    else:
        print(f"Loading metrics from: {metrics_file}")
        metrics = load_metrics_from_json(metrics_file)
    
    # Print summary table
    print_summary_table(metrics)
    
    # Generate plots
    print("Generating training curves...")
    plot_reranker_curves(metrics, args.output_dir, args.model_name)
    
    print(f"\n✓ All plots saved to: {args.output_dir}")
    print("✓ Done!")

if __name__ == '__main__':
    main()
