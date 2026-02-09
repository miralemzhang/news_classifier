"""
两阶段网页分类器
Stage 1: Embedding 召回 Top-2 候选
Stage 2: DeepSeek 从 Top-2 中选出 Top-1
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

from embedding_classifier import (
    QwenEmbeddingClassifier,
    load_labeled_data,
    LABELS,
)


@dataclass
class EmbeddingDeepSeekResult:
    """两阶段分类结果"""
    label: str                      # 最终分类标签（DeepSeek选择）
    confidence: float               # 置信度
    topk_candidates: List[str]      # Stage1 的 top-k 候选（送入DeepSeek的）
    topk_scores: Dict[str, float]   # Stage1 的 top-k 分数
    all_candidates: List[str]       # 所有类别按embedding分数排序
    all_scores: Dict[str, float]    # 所有类别的embedding分数
    thinking: str                   # DeepSeek 思考过程
    raw_output: str                 # DeepSeek 原始输出
    latency_ms: float               # 总耗时


# 类别描述 v1：详细列表格式
CATEGORY_DESCRIPTIONS_V1 = {
    "时政": [
        "国家领导人会议讲话，政府工作报告，政策法规发布",
        "两会新闻，党建工作，政府机构改革，干部任免，反腐倡廉",
        "选举投票，议会立法，政党政治，政府换届，政治选举",
        "外交访问，国际关系声明，政府间会谈，外交政策",
        "政府预算案讨论，财政政策辩论，议会预算审议",
    ],
    "经济": [
        "股票市场行情，基金理财，银行利率调整，金融市场波动",
        "企业财报，上市公司，投资并购，公司业绩，商业新闻",
        "房价走势，物价指数，GDP增长数据，经济增长率统计",
        "贸易进出口数据，国际贸易额，进出口统计",
        "消费零售，电商平台，金融产品，创业融资，市场分析",
    ],
    "军事": [
        "军队演习，武器装备，国防建设",
        "战争冲突，军事行动，武装冲突，军事打击，战场战况",
        "军舰航母，战斗机，导弹，坦克，军事科技，武器系统",
        "空袭轰炸，炮击袭击，导弹袭击，军事干预，军事打击行动",
        "军事部署，边境武装对峙，前线战况，部队调动",
    ],
    "社会": [
        "民生新闻，社会热点，公共事件，市民生活",
        "教育医疗，交通出行，社会保障，公共服务",
        "列车事故，车祸，空难，沉船，工业事故，安全事故",
        "犯罪案件，社会治安，法律诉讼，民事纠纷，刑事案件",
        "人道主义危机，饥荒，疫病，难民生存状况",
    ],
    "科技": [
        "人工智能，互联网科技，手机数码产品，科技公司",
        "科学研究，技术创新，航天航空，太空探索",
        "新能源汽车，芯片半导体，5G通信，电动车",
        "软件应用，云计算，大数据，网络安全",
        "机器人，无人机技术研发，智能设备，科技产品发布",
    ],
    "体育": [
        "足球篮球比赛，运动员，体育赛事，比赛结果",
        "奥运会，世界杯，NBA，CBA联赛，欧冠",
        "健身运动，马拉松，电子竞技，极限运动",
        "网球高尔夫，游泳田径，冬季运动",
        "体育明星，球队转会，体育评论，赛事分析",
    ],
    "娱乐": [
        "明星八卦，影视综艺，音乐演唱会，娱乐新闻",
        "电影电视剧，选秀节目，网红直播",
        "游戏动漫，时尚潮流，颁奖典礼",
        "娱乐圈，粉丝文化，偶像团体",
        "艺术展览，文艺演出，戏剧话剧，文化节目",
    ],
    "其他": [
        "生活百科，美食旅游，健康养生",
        "天气预报，星座运势，历史文化",
        "日常杂谈，情感故事，职场经验，个人博客",
        "宠物动物，家居装修，购物指南，生活服务",
        "广告软文，营销推广，品牌宣传",
    ],
}

# 类别描述 v2：简洁字符串格式
CATEGORY_DESCRIPTIONS_V2 = {
    "时政": "国家领导人、政府工作、政策法规、选举投票、外交关系、党建工作",
    "经济": "股票市场、企业财报、房价物价、贸易进出口、GDP增长、金融货币",
    "军事": "军队演习、武器装备、战争冲突、军事部署、国防建设",
    "社会": "民生新闻、教育医疗、交通事故、犯罪案件、社会热点",
    "科技": "人工智能、互联网科技、手机数码、科学研究、航天航空",
    "体育": "足球篮球、奥运会、世界杯、体育明星、运动赛事",
    "娱乐": "明星八卦、影视综艺、音乐演出、游戏动漫、娱乐圈",
    "其他": "生活百科、美食旅游、天气预报、广告营销、无法归类",
}

# 版本映射
CATEGORY_DOCS_VERSIONS = {
    "v1": CATEGORY_DESCRIPTIONS_V1,
    "v2": CATEGORY_DESCRIPTIONS_V2,
}


class EmbeddingDeepSeekClassifier:
    """Embedding + DeepSeek 两阶段分类器"""

    def __init__(
        self,
        embedding_model_path: str = None,
        deepseek_model_path: str = None,
        device: str = None,
        top_k: int = 2,
        doc_version: str = "v1",
    ):
        """
        初始化两阶段分类器

        Args:
            embedding_model_path: Embedding 模型路径
            deepseek_model_path: DeepSeek 模型路径
            device: 设备
            top_k: Stage1 召回的候选数量
            doc_version: 类别描述版本 (v1=详细列表, v2=简洁字符串)
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
        self.doc_version = doc_version
        self.category_docs = CATEGORY_DOCS_VERSIONS.get(doc_version, CATEGORY_DESCRIPTIONS_V1)
        print(f"  使用类别描述版本: {doc_version}")

        # Stage 1: 加载 Embedding 模型
        print("=" * 60)
        print(" Embedding + DeepSeek 两阶段分类器")
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

    def _build_deepseek_prompt(
        self,
        title: str,
        content: str,
        candidates: List[str],
    ) -> str:
        """构建 DeepSeek 判断 prompt"""
        # 获取类别描述文本（支持 v1 列表格式和 v2 字符串格式）
        def get_doc_text(label):
            doc = self.category_docs[label]
            if isinstance(doc, list):
                return "\n      - ".join(doc)  # v1: 列表 -> 换行格式
            return doc  # v2: 直接返回字符串

        # 候选类别描述
        candidate_lines = []
        for i, cat in enumerate(candidates):
            desc_str = get_doc_text(cat)
            if isinstance(self.category_docs[cat], list):
                candidate_lines.append(f"  {i+1}. {cat}:\n      - {desc_str}")
            else:
                candidate_lines.append(f"  {i+1}. {cat}: {desc_str}")
        candidate_desc = "\n".join(candidate_lines)

        prompt = f"""你是一个新闻分类专家。根据以下新闻内容，从给定的候选类别中选择最合适的一个。

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
    ) -> EmbeddingDeepSeekResult:
        """
        两阶段分类

        Args:
            title: 标题
            content: 内容
            url: URL

        Returns:
            EmbeddingDeepSeekResult: 分类结果
        """
        start_time = time.time()

        # Stage 1: Embedding 召回 Top-K
        emb_result = self.embedder.classify(title=title, content=content, url=url)

        # 获取所有候选（按分数排序）
        sorted_scores = sorted(emb_result.scores.items(), key=lambda x: -x[1])
        all_candidates = [label for label, _ in sorted_scores]
        all_scores = dict(sorted_scores)

        # Top-K 候选（送入 DeepSeek）
        top_k_candidates = all_candidates[:self.top_k]
        top_k_scores = {label: all_scores[label] for label in top_k_candidates}

        # Stage 2: DeepSeek 从候选中选择
        if len(top_k_candidates) <= 1:
            # 只有一个候选，直接返回
            final_label = top_k_candidates[0]
            thinking = ""
            raw_output = ""
        else:
            # 构建 prompt
            prompt = self._build_deepseek_prompt(title, content, top_k_candidates)

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

        return EmbeddingDeepSeekResult(
            label=final_label,
            confidence=top_k_scores.get(final_label, 0.5),
            topk_candidates=top_k_candidates,
            topk_scores=top_k_scores,
            all_candidates=all_candidates,
            all_scores=all_scores,
            thinking=thinking,
            raw_output=raw_output if len(top_k_candidates) > 1 else "",
            latency_ms=latency_ms,
        )

    def batch_classify(
        self,
        items: List[Dict],
    ) -> List[EmbeddingDeepSeekResult]:
        """批量分类"""
        results = []
        print(f"\n  {'='*60}")
        print(f"  Embedding + DeepSeek 两阶段分类 (共 {len(items)} 条)")
        print(f"  {'='*60}")

        for idx, item in enumerate(items):
            title = item.get("title", "")
            content = item.get("text", item.get("content", ""))
            url = item.get("articleLink", item.get("url", ""))

            result = self.classify(title=title, content=content, url=url)
            results.append(result)

            # 打印进度
            item_id = item.get("id", idx)
            topk_str = ", ".join(result.topk_candidates)
            print(f"  [{item_id:4d}] {title[:35]:35s}")
            print(f"         Top{len(result.topk_candidates)}: [{topk_str}] → {result.label}")

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
    save_path: str = None,
    load_path: str = None,
    doc_version: str = "v1",
):
    """
    训练并评估 Embedding + DeepSeek 分类器
    """
    print("=" * 70)
    print(" Embedding + DeepSeek 两阶段分类器 - 训练与评估")
    print(f" Stage 1: Embedding (Top-{top_k}) + Stage 2: DeepSeek (Top-1)")
    print("=" * 70)
    print(f"数据文件: {data_path}")
    print(f"训练样本: {train_count} 条")
    print(f"测试样本: {test_count} 条")
    print(f"类别描述版本: {doc_version}")
    print("=" * 70)

    # 加载数据
    print("\n[1/5] 加载数据...")
    all_data = load_labeled_data(data_path)
    print(f"  总计: {len(all_data)} 条")

    train_data = all_data[:train_count]
    test_data = all_data[train_count:train_count + test_count]
    print(f"  训练集: {len(train_data)} 条")
    print(f"  测试集: {len(test_data)} 条")

    # 初始化分类器
    print("\n[2/5] 初始化分类器...")
    classifier = EmbeddingDeepSeekClassifier(top_k=top_k, doc_version=doc_version)

    # 训练或加载 Embedding
    if load_path:
        print("\n[3/5] 加载已训练的类别嵌入...")
        classifier.load_category_matrix(load_path)
    else:
        print("\n[3/5] 训练 Embedding...")
        classifier.train_from_data(train_data)
        if save_path:
            classifier.save_category_matrix(save_path)

    # 测试
    print("\n[4/5] 测试...")
    results = classifier.batch_classify(test_data)

    # 评估
    print("\n[5/5] 评估...")

    # 统计各级 Top-K 召回和 DeepSeek 精选后准确率
    topk_recall_counts = {k: 0 for k in range(1, top_k + 1)}
    topk_deepseek_correct = {k: 0 for k in range(1, top_k + 1)}
    errors = []

    for item, result in zip(test_data, results):
        true_label = item.get("label")
        deepseek_pick = result.label

        # 统计各级 Top-K 召回
        for k in range(1, top_k + 1):
            top_k_candidates = result.all_candidates[:k]
            if true_label in top_k_candidates:
                topk_recall_counts[k] += 1

        # 统计各级 Top-K + DeepSeek 准确率
        # k=1: 直接用 embedding top-1
        if true_label == result.all_candidates[0]:
            topk_deepseek_correct[1] += 1

        # k>=2: 如果 DeepSeek 的选择在 top-k 内，用 DeepSeek；否则用 embedding top-1
        for k in range(2, top_k + 1):
            top_k_candidates = result.all_candidates[:k]
            if deepseek_pick in top_k_candidates:
                # DeepSeek 选择在候选范围内
                if true_label == deepseek_pick:
                    topk_deepseek_correct[k] += 1
            else:
                # DeepSeek 选择不在候选范围内，fallback 到 embedding top-1
                if true_label == result.all_candidates[0]:
                    topk_deepseek_correct[k] += 1

        # 记录 Top-K 未召回的错误样本
        if true_label not in result.topk_candidates:
            errors.append({
                "id": item.get("id"),
                "title": item.get("title", "")[:50],
                "true": true_label,
                "pred": deepseek_pick,
                "topk": result.topk_candidates,
            })

    # 各类别准确率（基于最终 DeepSeek 选择）
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

    # Top-K Embedding 召回率
    print("\n  【Embedding 召回率】")
    print("  " + "-" * 50)
    recall_header = "  "
    recall_values = "  "
    for k in range(1, top_k + 1):
        recall_rate = topk_recall_counts[k] / len(test_data) if test_data else 0
        recall_header += f"Top-{k:d}      "
        recall_values += f"{recall_rate:6.2%}    "
    print(recall_header)
    print(recall_values)

    # Top-K + DeepSeek 准确率
    print("\n  【Embedding + DeepSeek 准确率】")
    print("  " + "-" * 50)
    acc_header = "  "
    acc_values = "  "
    for k in range(1, top_k + 1):
        acc_rate = topk_deepseek_correct[k] / len(test_data) if test_data else 0
        if k == 1:
            acc_header += "Emb-Top1   "
        else:
            acc_header += f"Top{k}+DS    "
        acc_values += f"{acc_rate:6.2%}    "
    print(acc_header)
    print(acc_values)

    # 最终准确率
    final_accuracy = topk_deepseek_correct[top_k] / len(test_data) if test_data else 0
    print(f"\n  最终准确率 (Top-{top_k} + DeepSeek): {final_accuracy:.2%} ({topk_deepseek_correct[top_k]}/{len(test_data)})")

    print("\n  各类别准确率:")
    for label, stats in per_class_metrics.items():
        print(f"    {label}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    if errors:
        print(f"\n  Top-{top_k} 未召回的样本 (共 {len(errors)} 条):")
        for err in errors[:5]:
            print(f"    [{err['id']:4d}] {err['title'][:40]}")
            print(f"           真实: {err['true']}, Top{top_k}: {err['topk']}")

    # 保存结果
    result_dir = Path("/home/zzh/webpage-classification/data/results")
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_filename = f"embedding_deepseek_train{train_count}_topk{top_k}_{doc_version}_test{test_count}_{timestamp}.json"
    result_path = result_dir / result_filename

    # 构建结果数据
    topk_recall_rates = {f'top{k}_recall': topk_recall_counts[k] / len(test_data)
                         for k in range(1, top_k + 1)}
    topk_accuracy_rates = {f'top{k}_deepseek_accuracy': topk_deepseek_correct[k] / len(test_data)
                           for k in range(1, top_k + 1)}

    result_data = {
        'timestamp': timestamp,
        'config': {
            'method': 'Embedding + DeepSeek',
            'train_count': train_count,
            'test_count': test_count,
            'top_k': top_k,
            'doc_version': doc_version,
        },
        'overall': {
            'final_accuracy': topk_deepseek_correct[top_k] / len(test_data),
            'total': len(test_data),
            **topk_recall_rates,
            **topk_accuracy_rates,
        },
        'per_class': per_class_metrics,
        'errors': errors,
    }

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"\n  结果已保存至: {result_path}")
    print("=" * 70)

    final_accuracy = topk_deepseek_correct[top_k] / len(test_data) if test_data else 0
    final_recall = topk_recall_counts[top_k] / len(test_data) if test_data else 0
    return final_accuracy, final_recall, errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embedding + DeepSeek 两阶段分类器")
    parser.add_argument("--data", default="/home/zzh/webpage-classification/data/labeled/labeled.jsonl",
                       help="标注数据路径")
    parser.add_argument("--train", type=int, default=500, help="训练样本数")
    parser.add_argument("--test", type=int, default=100, help="测试样本数")
    parser.add_argument("--top-k", type=int, default=2, help="Stage1 召回数量")
    parser.add_argument("--doc-version", choices=["v1", "v2"], default="v1",
                       help="类别描述版本 (v1=详细列表, v2=简洁字符串)")
    parser.add_argument("--save", help="保存类别嵌入的路径")
    parser.add_argument("--load", help="加载已训练的类别嵌入")

    args = parser.parse_args()

    # 自动生成保存路径 (Embedding 矩阵与 top_k/doc_version 无关)
    if not args.load and not args.save:
        args.save = f"../../models/matrix_train{args.train}.pt"

    # 自动保存 log
    result_dir = Path("/home/zzh/webpage-classification/data/results")
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = result_dir / f"embedding_deepseek_train{args.train}_topk{args.top_k}_{args.doc_version}_test{args.test}_{timestamp}.log"
    logger = TeeLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    print(f"Log 保存至: {log_path}")

    train_and_evaluate(
        data_path=args.data,
        train_count=args.train,
        test_count=args.test,
        top_k=args.top_k,
        save_path=args.save,
        load_path=args.load,
        doc_version=args.doc_version,
    )
