"""
评估模块
计算准确率、生成混淆矩阵和图表

使用:
    python evaluate.py -p data/results/200to399_result_01.json -l data/labeled/top400_labeled.jsonl
    python evaluate.py -p data/results/200to399_result_01.json -l data/labeled/top400_labeled.jsonl --start 200 --end 399
"""

import json
import re
import argparse
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 8 类分类体系
LABELS = ["时政", "经济", "军事", "社会", "科技", "体育", "娱乐", "其他"]

# 默认文件路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_LABELED = PROJECT_ROOT / "data/labeled/top400_labeled.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/eval"


def load_json_file(file_path: str) -> list[dict]:
    """加载 JSON/JSONL/多行JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    first_char = content.strip()[0] if content.strip() else ''

    if first_char == '[':
        # JSON 数组格式
        return json.loads(content)
    elif first_char == '{':
        # 多行 JSON 对象格式
        objects = []
        depth = 0
        start = -1
        for i, char in enumerate(content):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    obj_str = content[start:i+1]
                    obj_str = re.sub(r',(\s*})', r'\1', obj_str)
                    try:
                        obj = json.loads(obj_str)
                        objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = -1
        return objects
    else:
        # 标准 JSONL 格式
        return [json.loads(line) for line in content.split('\n') if line.strip()]


def merge_predictions_and_labels(predictions: list[dict], labels: list[dict]) -> list[dict]:
    """合并预测结果和真实标签"""
    # 构建标签字典 (id -> label)
    label_map = {item.get('id'): item.get('label') for item in labels}

    results = []
    for pred in predictions:
        pred_id = pred.get('id')
        true_label = label_map.get(pred_id)

        if true_label is None:
            continue

        results.append({
            'id': pred_id,
            'title': pred.get('title', ''),
            'true_label': true_label,
            'predicted_label': pred.get('category', pred.get('predicted_label', '')),
            'confidence': pred.get('confidence', 0),
            'method': pred.get('method', ''),
        })

    return results


def calculate_accuracy(results: list[dict]) -> dict:
    """计算准确率"""
    correct = sum(1 for r in results if r.get("predicted_label") == r.get("true_label"))
    total = len(results)
    return {"accuracy": correct / total if total > 0 else 0, "correct": correct, "total": total}


def calculate_per_class_metrics(results: list[dict]) -> dict:
    """计算每个类别的指标"""
    metrics = {}
    for label in LABELS:
        tp = sum(1 for r in results if r.get("predicted_label") == label and r.get("true_label") == label)
        fp = sum(1 for r in results if r.get("predicted_label") == label and r.get("true_label") != label)
        fn = sum(1 for r in results if r.get("predicted_label") != label and r.get("true_label") == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return metrics


def generate_confusion_matrix(results: list[dict], output_path: str):
    """生成混淆矩阵图"""
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    label_to_idx = {label: i for i, label in enumerate(LABELS)}

    for r in results:
        true_idx = label_to_idx.get(r.get("true_label"), len(LABELS) - 1)
        pred_idx = label_to_idx.get(r.get("predicted_label"), len(LABELS) - 1)
        matrix[true_idx][pred_idx] += 1

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(len(LABELS)), LABELS, rotation=45, ha='right')
    plt.yticks(range(len(LABELS)), LABELS)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('混淆矩阵')

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            plt.text(j, i, str(matrix[i][j]), ha='center', va='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存: {output_path}")


def save_errors(results: list[dict], output_path: str):
    """保存错误样本"""
    errors = [r for r in results if r.get("predicted_label") != r.get("true_label")]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    print(f"错误样本已保存: {output_path} ({len(errors)} 条)")
    return errors


def evaluate(
    predictions_path: str,
    labels_path: str = None,
    output_dir: str = None,
    start_id: int = None,
    end_id: int = None,
):
    """
    评估分类结果

    Args:
        predictions_path: 预测结果文件路径
        labels_path: 标签文件路径
        output_dir: 输出目录
        start_id: 起始 id（包含）
        end_id: 结束 id（包含）
    """
    labels_path = Path(labels_path) if labels_path else DEFAULT_LABELED
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 从预测文件名生成输出文件名前缀
    pred_stem = Path(predictions_path).stem

    print("=" * 60)
    print(" 分类评估")
    print("=" * 60)
    print(f"预测文件: {predictions_path}")
    print(f"标签文件: {labels_path}")
    if start_id is not None or end_id is not None:
        print(f"ID 范围: {start_id if start_id is not None else '起始'} - {end_id if end_id is not None else '结束'}")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] 加载数据...")
    predictions = load_json_file(predictions_path)
    labels = load_json_file(str(labels_path))
    print(f"  预测结果: {len(predictions)} 条")
    print(f"  标签数据: {len(labels)} 条")

    # 合并
    print("\n[2/4] 合并预测和标签...")
    results = merge_predictions_and_labels(predictions, labels)

    # 根据 id 范围过滤
    if start_id is not None or end_id is not None:
        original_count = len(results)
        results = [
            r for r in results
            if (start_id is None or r.get('id', -1) >= start_id) and
               (end_id is None or r.get('id', -1) <= end_id)
        ]
        print(f"  过滤后: {len(results)} 条 (原 {original_count} 条)")
    else:
        print(f"  合并后: {len(results)} 条")

    if not results:
        print("  错误: 没有可评估的数据!")
        return

    # 计算指标
    print("\n[3/4] 计算评估指标...")
    accuracy = calculate_accuracy(results)
    per_class = calculate_per_class_metrics(results)

    # 按方法统计
    method_stats = Counter(r.get('method') for r in results)
    method_accuracy = {}
    for method in method_stats:
        method_results = [r for r in results if r.get('method') == method]
        method_accuracy[method] = calculate_accuracy(method_results)

    # 输出结果
    print("\n[4/4] 生成报告...")

    # 生成混淆矩阵
    generate_confusion_matrix(results, str(output_dir / f"{pred_stem}_confusion.png"))

    # 保存错误样本
    errors = save_errors(results, str(output_dir / f"{pred_stem}_errors.json"))

    # 打印报告
    print("\n" + "=" * 60)
    print(" 评估结果")
    print("=" * 60)
    print(f"\n总体准确率: {accuracy['accuracy']:.2%} ({accuracy['correct']}/{accuracy['total']})")

    print("\n按分类方法:")
    for method, stats in method_accuracy.items():
        method_name = {'rule': '规则', 'model': '模型'}.get(method, method)
        print(f"  {method_name}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    print("\n各类别指标:")
    print("-" * 50)
    for label, m in per_class.items():
        if m['support'] > 0:
            print(f"  {label}: P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} (n={m['support']})")

    # 保存完整报告
    report = {
        "predictions_file": str(predictions_path),
        "labels_file": str(labels_path),
        "id_range": {"start": start_id, "end": end_id},
        "overall_accuracy": accuracy,
        "method_accuracy": method_accuracy,
        "per_class_metrics": per_class,
        "error_count": len(errors),
    }

    report_path = output_dir / f"{pred_stem}_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n完整报告已保存: {report_path}")

    # 更新 summary.json
    summary_path = output_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        except:
            summary = {}

    summary[pred_stem] = {
        "accuracy": accuracy["accuracy"],
        "correct": accuracy["correct"],
        "total": accuracy["total"],
        "id_range": f"{start_id}-{end_id}" if start_id is not None else "all",
        "error_count": len(errors),
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"汇总已更新: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="评估分类结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python evaluate.py -p data/results/0to199_result_01.json --start 0 --end 199
  python evaluate.py -p data/results/200to399_result_01.json --start 200 --end 399
  python evaluate.py -p data/results/200to399_result_01.json   # 自动根据预测文件的 id 范围评估

标签文件默认: data/labeled/top400_labeled.jsonl
        """
    )
    parser.add_argument("-p", "--predictions", required=True, help="预测结果文件路径")
    parser.add_argument("-l", "--labels", help="标签文件路径 (默认: data/labeled/top400_labeled.jsonl)")
    parser.add_argument("-o", "--output", help="输出目录 (默认: data/eval)")
    parser.add_argument("--start", type=int, help="起始 id（包含）")
    parser.add_argument("--end", type=int, help="结束 id（包含）")

    args = parser.parse_args()

    evaluate(
        predictions_path=args.predictions,
        labels_path=args.labels,
        output_dir=args.output,
        start_id=args.start,
        end_id=args.end,
    )


if __name__ == '__main__':
    main()
