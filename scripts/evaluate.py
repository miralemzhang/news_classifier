#!/usr/bin/env python3
"""
Evaluation script for webpage classification results.

Usage:
    python scripts/evaluate.py <prediction_file> [--ground-truth <labeled_file>] [--output-dir <output_dir>]

Examples:
    python scripts/evaluate.py data/results/top200_result_06.json
    python scripts/evaluate.py data/results/top200_result_06.json --ground-truth data/labeled/top100_labeled.jsonl
    python scripts/evaluate.py data/results/top200_result_06.json --output-dir data/eval
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(file_path):
    """Load data from JSONL file (supports multi-line formatted JSON objects)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    objects = re.findall(pattern, content, re.DOTALL)

    data = []
    for obj_str in objects:
        try:
            data.append(json.loads(obj_str))
        except json.JSONDecodeError:
            continue
    return data


def load_json(file_path):
    """Load data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_label(label):
    """Normalize label for comparison (handle 其它 vs 其他)."""
    if label in ['其它', '其他']:
        return '其它'
    return label


def evaluate(predictions, ground_truth, max_eval=100):
    """
    Evaluate predictions against ground truth.

    Args:
        predictions: List of prediction dicts with 'id' and 'category'
        ground_truth: List of ground truth dicts with 'id' and 'label'
        max_eval: Maximum number of items to evaluate (default 100, first 100 labeled)

    Returns:
        Dictionary with evaluation metrics
    """
    # Create lookup by id
    pred_by_id = {p['id']: p for p in predictions}
    truth_by_id = {t['id']: t for t in ground_truth}

    # Only evaluate items that exist in both and within max_eval range
    eval_ids = sorted([id for id in pred_by_id.keys() if id in truth_by_id and id < max_eval])

    if not eval_ids:
        raise ValueError("No matching IDs found between predictions and ground truth")

    # Collect results
    correct = 0
    errors = []
    y_true = []
    y_pred = []

    for id in eval_ids:
        pred = pred_by_id[id]
        truth = truth_by_id[id]

        pred_label = normalize_label(pred['category'])
        true_label = normalize_label(truth['label'])

        y_true.append(true_label)
        y_pred.append(pred_label)

        if pred_label == true_label:
            correct += 1
        else:
            errors.append({
                'id': id,
                'title': pred.get('title', truth.get('title', '')),
                'true_label': truth['label'],
                'predicted_label': pred['category'],
                'confidence': pred.get('confidence', 0),
                'method': pred.get('method', 'unknown')
            })

    total = len(eval_ids)
    accuracy = correct / total if total > 0 else 0

    # Calculate per-class metrics
    labels = ['时政', '经济', '军事', '社会', '科技', '体育', '娱乐', '其它']
    per_class = {}

    for label in labels:
        # True positives, false positives, false negatives
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_class[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'errors': errors,
        'per_class': per_class,
        'y_true': y_true,
        'y_pred': y_pred,
        'labels': labels
    }


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """Generate and save confusion matrix plot."""
    # Build confusion matrix
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            matrix[label_to_idx[t]][label_to_idx[p]] += 1

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='Blues')

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, matrix[i, j], ha="center", va="center",
                          color="white" if matrix[i, j] > matrix.max()/2 else "black")

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate webpage classification results')
    parser.add_argument('prediction_file', help='Path to prediction JSON file')
    parser.add_argument('--ground-truth', '-g',
                       default='data/labeled/top200_labeled.jsonl',
                       help='Path to ground truth labeled file (default: data/labeled/top200_labeled.jsonl)')
    parser.add_argument('--output-dir', '-o',
                       default='data/eval',
                       help='Output directory for evaluation results (default: data/eval)')
    parser.add_argument('--max-eval', '-n', type=int, default=200,
                       help='Maximum number of items to evaluate (default: 200)')

    args = parser.parse_args()

    # Load data
    pred_path = Path(args.prediction_file)
    if not pred_path.exists():
        print(f"Error: Prediction file not found: {pred_path}")
        return 1

    truth_path = Path(args.ground_truth)
    if not truth_path.exists():
        print(f"Error: Ground truth file not found: {truth_path}")
        return 1

    print(f"Loading predictions from: {pred_path}")
    predictions = load_json(pred_path) if pred_path.suffix == '.json' else load_jsonl(pred_path)

    print(f"Loading ground truth from: {truth_path}")
    ground_truth = load_jsonl(truth_path)

    # Evaluate
    print(f"Evaluating (max {args.max_eval} items)...")
    results = evaluate(predictions, ground_truth, args.max_eval)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
    print(f"\nPer-class metrics:")
    print(f"{'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"{'-'*48}")
    for label in results['labels']:
        m = results['per_class'][label]
        if m['support'] > 0:
            print(f"{label:<8} {m['precision']:>10.2%} {m['recall']:>10.2%} {m['f1']:>10.2%} {m['support']:>10}")

    print(f"\nErrors: {len(results['errors'])}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = pred_path.stem

    # Save report
    report = {
        'model': model_name,
        'eval_count': results['total'],
        'overall_accuracy': {
            'accuracy': results['accuracy'],
            'correct': results['correct'],
            'total': results['total']
        },
        'per_class_metrics': results['per_class']
    }

    report_path = output_dir / f"{model_name}_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to: {report_path}")

    # Save errors
    errors_path = output_dir / f"{model_name}_errors.json"
    with open(errors_path, 'w', encoding='utf-8') as f:
        json.dump(results['errors'], f, ensure_ascii=False, indent=2)
    print(f"Errors saved to: {errors_path}")

    # Save confusion matrix
    cm_path = output_dir / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix(results['y_true'], results['y_pred'], results['labels'], cm_path)

    # Update summary
    summary_path = output_dir / 'summary.json'
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    else:
        summary = {}

    summary[model_name] = {
        'accuracy': results['accuracy'],
        'correct': results['correct'],
        'total': results['total']
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary updated: {summary_path}")

    return 0


if __name__ == '__main__':
    exit(main())
