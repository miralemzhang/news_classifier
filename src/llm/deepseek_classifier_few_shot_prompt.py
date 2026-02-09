"""
两阶段网页分类器 (Few-shot Prompt 版本)
Stage 1: Embedding 召回 Top-2 候选
Stage 2: DeepSeek + Few-shot Prompt 从 Top-2 中选出 Top-1
"""

import torch
import time
import re
import json
import sys
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

from embedding_classifier import (
    QwenEmbeddingClassifier,
    load_labeled_data,
    LABELS,
)


class TeeLogger:
    """同时输出到终端和文件"""
    def __init__(self, filename, mode='w'):
        self.terminal = sys.stdout
        self.log = open(filename, mode, encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


@dataclass
class FewShotResult:
    """Few-shot 分类结果"""
    label: str                      # 最终分类标签
    confidence: float               # 置信度
    top2_candidates: List[str]      # Stage1 的 top2 候选
    top2_scores: Dict[str, float]   # Stage1 的 top2 分数
    thinking: str                   # DeepSeek 思考过程
    raw_output: str                 # DeepSeek 原始输出
    latency_ms: float               # 总耗时


# 类别描述 (V2 简洁版)
CATEGORY_DESCRIPTIONS = {
    "时政": "国家领导人、政府工作、政策法规、选举投票、外交关系、党建工作",
    "经济": "股票市场、企业财报、房价物价、贸易进出口、GDP增长、金融货币",
    "军事": "军队演习、武器装备、战争冲突、军事部署、国防建设",
    "社会": "民生新闻、教育医疗、交通事故、犯罪案件、社会热点",
    "科技": "人工智能、互联网科技、手机数码、科学研究、航天航空",
    "体育": "足球篮球、奥运会、世界杯、体育明星、运动赛事",
    "娱乐": "明星八卦、影视综艺、音乐演出、游戏动漫、娱乐圈",
    "其他": "生活百科、美食旅游、天气预报、广告营销、无法归类",
}


class FewShotClassifier:
    """Embedding + DeepSeek Few-shot 两阶段分类器"""

    def __init__(
        self,
        embedding_model_path: str = None,
        deepseek_model_path: str = None,
        device: str = None,
        top_k: int = 2,
        few_shot_examples: Dict[str, List[Dict]] = None,
    ):
        """
        初始化分类器

        Args:
            embedding_model_path: Embedding 模型路径
            deepseek_model_path: DeepSeek 模型路径
            device: 设备
            top_k: Stage1 召回的候选数量
            few_shot_examples: Few-shot 示例 {类别: [{"title": ..., "content": ...}, ...]}
        """
        base_dir = Path(__file__).parent.parent.parent

        if embedding_model_path is None:
            embedding_model_path = str(base_dir / "models" / "Qwen3-VL-Embedding-8B")

        if deepseek_model_path is None:
            deepseek_model_path = str(base_dir / "models" / "DeepSeek-R1-Distill-Qwen-7B")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.top_k = top_k
        self.few_shot_examples = few_shot_examples or {}

        # Stage 1: 加载 Embedding 模型
        print("=" * 60)
        print(" Embedding + DeepSeek Few-shot 两阶段分类器")
        print("=" * 60)
        print("\n[Stage 1] 加载 Embedding 模型...")
        self.embedder = QwenEmbeddingClassifier(
            model_path=embedding_model_path,
            device=device,
            use_template=True,
        )

        # Stage 2: 加载 DeepSeek 模型
        print("\n[Stage 2] 加载 DeepSeek 模型...")
        self._load_deepseek(deepseek_model_path)

        print("\n" + "=" * 60)
        print(" 初始化完成")
        print("=" * 60)

    def _load_deepseek(self, model_path: str):
        """加载 DeepSeek 模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.deepseek = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True
        )
        self.deepseek.eval()
        print(f"  DeepSeek 加载完成: {model_path}")

    def train_from_data(self, train_data: List[Dict], batch_size: int = 32):
        """用标注数据训练 Embedding 模型"""
        return self.embedder.train_from_data(train_data, batch_size)

    def build_few_shot_examples_from_data(
        self,
        train_data: List[Dict],
        examples_per_class: int = 2,
    ):
        """
        从训练数据中自动选择 few-shot 示例
        选择每个类别中标题最短且最具代表性的样本

        Args:
            train_data: 训练数据
            examples_per_class: 每个类别的示例数量
        """
        print(f"\n  自动选择 Few-shot 示例 (每类 {examples_per_class} 条)...")

        # 按类别分组
        by_label = {}
        for item in train_data:
            label = item.get("label")
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(item)

        # 每个类别选择标题最短的几个作为示例（通常标题短的更典型）
        self.few_shot_examples = {}
        for label in LABELS:
            if label in by_label:
                # 按标题长度排序，选择较短的
                sorted_items = sorted(by_label[label], key=lambda x: len(x.get("title", "")))
                selected = sorted_items[:examples_per_class]
                self.few_shot_examples[label] = [
                    {"title": item.get("title", ""), "content": item.get("text", "")[:100]}
                    for item in selected
                ]
                print(f"    {label}: {[item['title'][:20] + '...' for item in self.few_shot_examples[label]]}")

    def _build_few_shot_prompt(
        self,
        title: str,
        content: str,
        candidates: List[str],
    ) -> str:
        """构建 Few-shot Prompt"""

        # 构建示例部分（只用候选类别的示例）
        examples_text = ""
        for i, cat in enumerate(candidates):
            if cat in self.few_shot_examples:
                for ex in self.few_shot_examples[cat]:
                    examples_text += f"【示例】标题：{ex['title'][:50]} → {cat}\n"

        # 候选类别描述
        candidate_desc = "\n".join([
            f"  {i+1}. {cat}: {CATEGORY_DESCRIPTIONS[cat]}"
            for i, cat in enumerate(candidates)
        ])

        prompt = f"""你是一个新闻分类专家。请参考以下示例，从候选类别中选择最合适的一个。

{examples_text}
【新闻标题】
{title}

【新闻内容】
{content[:500] if content else "无"}

【候选类别】
{candidate_desc}

请仔细分析新闻内容，选择最合适的类别。只需回答类别名称，如"时政"或"经济"。

答案："""

        return prompt

    def _parse_deepseek_output(self, output: str, candidates: List[str]) -> tuple:
        """解析 DeepSeek 输出"""
        # 提取 thinking
        thinking = ""
        think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()

        # 在输出中查找候选类别
        for cat in candidates:
            if cat in output:
                return cat, thinking

        # 默认返回第一个候选
        return candidates[0], thinking

    def classify(
        self,
        title: str,
        content: str = "",
        url: str = "",
    ) -> FewShotResult:
        """两阶段分类"""
        start_time = time.time()

        # Stage 1: Embedding 召回 Top-K
        emb_result = self.embedder.classify(title=title, content=content, url=url)

        # 获取 Top-K 候选
        sorted_scores = sorted(emb_result.scores.items(), key=lambda x: -x[1])
        top_k_candidates = [label for label, _ in sorted_scores[:self.top_k]]
        top_k_scores = {label: score for label, score in sorted_scores[:self.top_k]}

        # Stage 2: DeepSeek 从候选中选择
        if len(top_k_candidates) <= 1:
            final_label = top_k_candidates[0]
            thinking = ""
            raw_output = ""
        else:
            # 构建 Few-shot Prompt
            prompt = self._build_few_shot_prompt(title, content, top_k_candidates)

            # DeepSeek 推理
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.deepseek.generate(
                    input_ids,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            raw_output = self.tokenizer.decode(
                outputs[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )

            final_label, thinking = self._parse_deepseek_output(raw_output, top_k_candidates)

        latency_ms = (time.time() - start_time) * 1000

        return FewShotResult(
            label=final_label,
            confidence=top_k_scores.get(final_label, 0.5),
            top2_candidates=top_k_candidates,
            top2_scores=top_k_scores,
            thinking=thinking,
            raw_output=raw_output if len(top_k_candidates) > 1 else "",
            latency_ms=latency_ms,
        )

    def batch_classify(self, items: List[Dict]) -> List[FewShotResult]:
        """批量分类"""
        results = []
        print(f"\n  {'='*60}")
        print(f"  Embedding + DeepSeek Few-shot 分类 (共 {len(items)} 条)")
        print(f"  {'='*60}")

        for idx, item in enumerate(items):
            title = item.get("title", "")
            content = item.get("text", item.get("content", ""))
            url = item.get("articleLink", item.get("url", ""))

            result = self.classify(title=title, content=content, url=url)
            results.append(result)

            # 打印进度
            item_id = item.get("id", idx)
            top2_str = ", ".join(result.top2_candidates)
            print(f"  [{item_id:4d}] {title[:35]:35s}")
            print(f"         Top2: [{top2_str}] → {result.label}")

            if (idx + 1) % 10 == 0:
                print(f"  --- 进度: {idx + 1}/{len(items)} ---")

        print(f"  {'='*60}\n")
        return results

    def save_category_matrix(self, path: str):
        """保存 Embedding 类别矩阵"""
        self.embedder.save_category_matrix(path)

    def load_category_matrix(self, path: str):
        """加载 Embedding 类别矩阵"""
        self.embedder.load_category_matrix(path)


def train_and_evaluate(
    data_path: str,
    train_count: int = 500,
    test_count: int = 100,
    top_k: int = 2,
    examples_per_class: int = 2,
    save_path: str = None,
    load_path: str = None,
):
    """训练并评估 Few-shot 分类器"""
    print("=" * 70)
    print(" Embedding + DeepSeek Few-shot 分类器 - 训练与评估")
    print(f" Stage 1: Embedding (Top-{top_k}) + Stage 2: DeepSeek Few-shot (Top-1)")
    print("=" * 70)
    print(f"数据文件: {data_path}")
    print(f"训练样本: {train_count} 条")
    print(f"测试样本: {test_count} 条")
    print(f"每类示例: {examples_per_class} 条")
    print("=" * 70)

    # 加载数据
    print("\n[1/6] 加载数据...")
    all_data = load_labeled_data(data_path)
    print(f"  总计: {len(all_data)} 条")

    train_data = all_data[:train_count]
    test_data = all_data[train_count:train_count + test_count]
    print(f"  训练集: {len(train_data)} 条")
    print(f"  测试集: {len(test_data)} 条")

    # 初始化分类器
    print("\n[2/6] 初始化分类器...")
    classifier = FewShotClassifier(top_k=top_k)

    # 构建 Few-shot 示例
    print("\n[3/6] 构建 Few-shot 示例...")
    classifier.build_few_shot_examples_from_data(train_data, examples_per_class)

    # 训练或加载 Embedding
    if load_path:
        print("\n[4/6] 加载已训练的类别嵌入...")
        classifier.load_category_matrix(load_path)
    else:
        print("\n[4/6] 训练 Embedding...")
        classifier.train_from_data(train_data)
        if save_path:
            classifier.save_category_matrix(save_path)

    # 测试
    print("\n[5/6] 测试...")
    results = classifier.batch_classify(test_data)

    # 评估
    print("\n[6/6] 评估...")
    correct = 0
    topk_recall = 0
    emb_top1_correct = 0
    errors = []

    for item, result in zip(test_data, results):
        true_label = item.get("label")
        pred_label = result.label
        emb_top1 = result.top2_candidates[0]

        if true_label == pred_label:
            correct += 1

        if true_label == emb_top1:
            emb_top1_correct += 1

        if true_label in result.top2_candidates:
            topk_recall += 1
        else:
            errors.append({
                "id": item.get("id"),
                "title": item.get("title", "")[:50],
                "true": true_label,
                "pred": pred_label,
                "top2": result.top2_candidates,
            })

    total = len(test_data)
    accuracy = correct / total if total else 0
    emb_top1_acc = emb_top1_correct / total if total else 0
    recall_at_k = topk_recall / total if total else 0

    # 各类别准确率
    per_class_metrics = {}
    for label in LABELS:
        label_items = [(item, result) for item, result in zip(test_data, results)
                       if item.get("label") == label]
        if label_items:
            label_correct = sum(1 for item, result in label_items
                               if item.get("label") == result.label)
            per_class_metrics[label] = {
                'accuracy': label_correct / len(label_items),
                'correct': label_correct,
                'total': len(label_items),
            }

    # 打印结果
    print("\n" + "=" * 70)
    print(" 评估结果")
    print("=" * 70)
    print(f"\n  ┌─────────────────┬────────────┬────────────┐")
    print(f"  │   指标          │   准确率   │   召回率   │")
    print(f"  ├─────────────────┼────────────┼────────────┤")
    print(f"  │ Embedding Top-1 │ {emb_top1_acc:>8.2%}   │ {emb_top1_acc:>8.2%}   │")
    print(f"  │ + DeepSeek FS   │ {accuracy:>8.2%}   │ {recall_at_k:>8.2%}   │")
    print(f"  └─────────────────┴────────────┴────────────┘")

    print(f"\n  详细数据:")
    print(f"    - Embedding Top-1 正确: {emb_top1_correct}/{total}")
    print(f"    - Few-shot 最终正确: {correct}/{total}")
    print(f"    - Top-{top_k} 召回: {topk_recall}/{total}")
    print(f"    - 提升: {accuracy - emb_top1_acc:+.2%}")

    print("\n  各类别准确率:")
    for label, stats in per_class_metrics.items():
        print(f"    {label}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    if errors:
        print(f"\n  Top-{top_k} 未召回的样本 (共 {len(errors)} 条):")
        for err in errors[:5]:
            print(f"    [{err['id']:4d}] {err['title'][:40]}")
            print(f"           真实: {err['true']}, Top{top_k}: {err['top2']}")

    # 保存结果
    result_dir = Path("/home/zzh/webpage-classification/data/results")
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_filename = f"deepseek_fewshot_train{train_count}_test{test_count}_{timestamp}.json"
    result_path = result_dir / result_filename

    result_data = {
        'timestamp': timestamp,
        'config': {
            'method': 'Embedding + DeepSeek Few-shot',
            'train_count': train_count,
            'test_count': test_count,
            'top_k': top_k,
            'examples_per_class': examples_per_class,
        },
        'overall': {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'emb_top1_accuracy': emb_top1_acc,
            f'top{top_k}_recall': recall_at_k,
        },
        'per_class': per_class_metrics,
        'errors': errors,
    }

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"\n  结果已保存至: {result_path}")
    print("=" * 70)

    return accuracy, recall_at_k, errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embedding + DeepSeek Few-shot 分类器")
    parser.add_argument("--data", default="/home/zzh/webpage-classification/data/labeled/labeled.jsonl",
                       help="标注数据路径")
    parser.add_argument("--train", type=int, default=500, help="训练样本数")
    parser.add_argument("--test", type=int, default=100, help="测试样本数")
    parser.add_argument("--top-k", type=int, default=2, help="Stage1 召回数量")
    parser.add_argument("--examples", type=int, default=2, help="每类 Few-shot 示例数")
    parser.add_argument("--save", help="保存类别嵌入的路径")
    parser.add_argument("--load", help="加载已训练的类别嵌入")

    args = parser.parse_args()

    # 自动生成保存路径 (Embedding 矩阵与 top_k 无关)
    if not args.load and not args.save:
        args.save = f"../../models/matrix_train{args.train}.pt"

    # 自动保存 log
    result_dir = Path("/home/zzh/webpage-classification/data/results")
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = result_dir / f"deepseek_fewshot_train{args.train}_test{args.test}_{timestamp}.log"
    logger = TeeLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    print(f"Log 保存至: {log_path}")

    train_and_evaluate(
        data_path=args.data,
        train_count=args.train,
        test_count=args.test,
        top_k=args.top_k,
        examples_per_class=args.examples,
        save_path=args.save,
        load_path=args.load,
    )
