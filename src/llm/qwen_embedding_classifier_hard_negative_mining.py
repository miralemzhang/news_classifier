"""
基于负样本挖掘 (Hard Negative Mining) 的网页分类器

原理：
1. 分析容易混淆的类别对
2. 在类别模板中加入区分说明
3. 训练时对难样本增加权重

容易混淆的类别对：
- 时政 ↔ 军事: 都涉及国家层面，但时政是政策，军事是军队
- 时政 ↔ 社会: 都涉及民生，但时政是政府视角，社会是民间视角
- 其他 ↔ 社会: 边界模糊，需要明确"其他"的定义
"""

import torch
import torch.nn.functional as F
import time
import json
import re
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from embedding_classifier import (
    QwenEmbeddingClassifier,
    EmbeddingClassifyResult,
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


# ═══════════════════════════════════════════════════════════════════════════
# 负样本挖掘策略配置
# ═══════════════════════════════════════════════════════════════════════════

# 已知容易混淆的类别对及其区分要点
CONFUSING_PAIRS = {
    ("时政", "军事"): {
        "时政_exclude": "军事行动、战争冲突、武器装备、军队演习",
        "军事_exclude": "政府工作、政策法规、外交谈判、党建工作",
        "distinction": "时政关注政策制定与政府行为，军事关注武装力量与战争冲突",
    },
    ("时政", "社会"): {
        "时政_exclude": "普通民生事件、社会热点、民间纠纷、事故案件",
        "社会_exclude": "国家领导人、政府政策、官方声明、党政机关",
        "distinction": "时政以政府/国家为主体，社会以民众/社会为主体",
    },
    ("其他", "社会"): {
        "其他_exclude": "社会新闻、民生热点、公共事件、案件事故",
        "社会_exclude": "生活百科、美食旅游、广告营销、无明确新闻价值",
        "distinction": "社会是有新闻价值的公共事件，其他是非新闻性内容",
    },
    ("经济", "科技"): {
        "经济_exclude": "科技产品发布、技术研发、AI/芯片技术",
        "科技_exclude": "公司财报、股价涨跌、投资并购、市场分析",
        "distinction": "经济侧重资金流动与市场，科技侧重技术创新与产品",
    },
    ("社会", "经济"): {
        "社会_exclude": "企业经营、股市行情、经济政策、商业活动",
        "经济_exclude": "社会民生、公共事件、犯罪案件、事故灾难",
        "distinction": "经济关注商业与金融，社会关注公共生活与民生",
    },
}

# 基础类别描述 (RERANKER_DOCS_V2)
CATEGORY_DOCS = {
    "时政": "国家领导人会议讲话，政府工作报告；两会新闻，党建工作，政府机构改革；选举投票，议会立法；外交访问，国际关系声明；政府预算案讨论；党和国家的重大方针政策；政府会议、文件、公报；习近平等党和国家领导人，新时代新思想；中国特色社会主义",
    "经济": "股票市场行情，基金理财，银行利率；企业财报，上市公司，投资并购；房价走势，物价指数，GDP增长；贸易进出口数据；消费零售，电商平台；汇率、交易、证券、财经",
    "军事": "军队演习，武器装备，国防建设；战争冲突，军事行动，武装冲突；军舰航母，战斗机，导弹；空袭轰炸，炮击袭击；军事部署，边境武装对峙",
    "社会": "民生新闻，社会热点，公共事件，社会保障、社会福利、人民利益；教育医疗，交通出行，社会保障；列车事故，车祸，空难；犯罪案件，社会治安；人道主义危机，饥荒",
    "科技": "人工智能（AI），互联网科技，手机数码，信息技术；科学研究，科技奖项，技术创新，航天航空；新能源汽车，芯片半导体；软件应用，云计算；机器人，无人机技术",
    "体育": "足球篮球比赛，运动员，体育赛事；奥运会，世界杯，NBA；健身运动，马拉松；网球、高尔夫、滑雪；体育明星，球队转会",
    "娱乐": "明星八卦，影视综艺，音乐会、演唱会、音乐剧、表演节目；电影电视剧，选秀节目；游戏动漫，时尚潮流；娱乐圈，粉丝文化；艺术展览，文艺演出；综艺、真人秀",
    "其他": "生活百科，美食旅游；天气预报，星座运势；日常杂谈，情感故事；宠物动物，家居装修；广告、营销、软文，营销推广",
}

# 混淆类别的排除说明（Hard Negative Mining 核心）
EXCLUSION_NOTES = {
    "时政": "（注：不包括军事行动、武装冲突；不包括普通社会民生事件、事故案件）",
    "军事": "（注：涉及武装力量、战争、武器；不包括纯政治外交谈判、政府声明）",
    "社会": "（注：有新闻价值的公共事件；不包括政府政策发布；不包括生活百科、广告软文）",
    "经济": "（注：侧重资金、市场、投资；不包括纯技术研发、科技产品发布）",
    "科技": "（注：侧重技术、产品、研发；不包括公司财报、股价分析、投资并购）",
    "其他": "（注：非新闻性内容、生活服务类、广告营销；不包括社会民生新闻）",
}

# 增强版类别模板：基础描述 + 排除说明
CATEGORY_TEMPLATES_ENHANCED = {
    "时政": [
        CATEGORY_DOCS["时政"],
        CATEGORY_DOCS["时政"] + EXCLUSION_NOTES["时政"],
    ],
    "经济": [
        CATEGORY_DOCS["经济"],
        CATEGORY_DOCS["经济"] + EXCLUSION_NOTES["经济"],
    ],
    "军事": [
        CATEGORY_DOCS["军事"],
        CATEGORY_DOCS["军事"] + EXCLUSION_NOTES["军事"],
    ],
    "社会": [
        CATEGORY_DOCS["社会"],
        CATEGORY_DOCS["社会"] + EXCLUSION_NOTES["社会"],
    ],
    "科技": [
        CATEGORY_DOCS["科技"],
        CATEGORY_DOCS["科技"] + EXCLUSION_NOTES["科技"],
    ],
    "体育": [
        CATEGORY_DOCS["体育"],
    ],
    "娱乐": [
        CATEGORY_DOCS["娱乐"],
    ],
    "其他": [
        CATEGORY_DOCS["其他"],
        CATEGORY_DOCS["其他"] + EXCLUSION_NOTES["其他"],
    ],
}


@dataclass
class HardNegativeResult:
    """分类结果（含难样本分析）"""
    label: str
    confidence: float
    scores: Dict[str, float]
    latency_ms: float
    # 难样本分析
    is_hard_negative: bool  # 是否为难样本
    confusing_pair: Tuple[str, str] = None  # 混淆的类别对
    margin: float = 0.0  # Top1 和 Top2 的分数差


class HardNegativeMiningClassifier:
    """
    基于负样本挖掘的分类器

    核心策略：
    1. 使用增强版类别模板（包含排除说明）
    2. 训练时对混淆类别对中的样本增加权重
    3. 分类时检测并标记难样本
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        hard_negative_weight: float = 2.0,
        margin_threshold: float = 0.05,
    ):
        """
        初始化分类器

        Args:
            model_path: 模型路径
            device: 设备
            hard_negative_weight: 难样本在训练时的权重倍数
            margin_threshold: 判定为难样本的 margin 阈值
        """
        base_dir = Path(__file__).parent.parent.parent

        if model_path is None:
            model_path = str(base_dir / "models" / "Qwen3-VL-Embedding-8B")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.hard_negative_weight = hard_negative_weight
        self.margin_threshold = margin_threshold

        # 构建混淆类别集合（用于快速查找）
        self.confusing_categories: Set[str] = set()
        self.confusing_pairs_set: Set[Tuple[str, str]] = set()
        for pair in CONFUSING_PAIRS.keys():
            self.confusing_categories.add(pair[0])
            self.confusing_categories.add(pair[1])
            self.confusing_pairs_set.add(pair)
            self.confusing_pairs_set.add((pair[1], pair[0]))  # 双向

        print("=" * 60)
        print(" Hard Negative Mining 分类器")
        print("=" * 60)
        print(f"难样本权重: {hard_negative_weight}x")
        print(f"Margin 阈值: {margin_threshold}")
        print(f"混淆类别对: {len(CONFUSING_PAIRS)} 对")
        for pair in CONFUSING_PAIRS.keys():
            print(f"  - {pair[0]} ↔ {pair[1]}")

        # 加载基础 embedding 分类器
        print("\n加载 Embedding 模型...")
        self.embedder = QwenEmbeddingClassifier(
            model_path=model_path,
            device=device,
            use_template=False,  # 不使用默认模板，我们用增强版
        )

        # 初始化增强版类别嵌入
        self._init_enhanced_embeddings()

        print("\n初始化完成!")
        print("=" * 60)

    def _init_enhanced_embeddings(self):
        """使用增强版模板初始化类别嵌入"""
        print("\n计算增强版类别嵌入...")

        for category, templates in CATEGORY_TEMPLATES_ENHANCED.items():
            # 为每个类别的所有模板计算嵌入
            inputs = [{"text": t, "instruction": "表示这段文本的类别特征。"} for t in templates]
            embeddings = self.embedder.embedder.process(inputs)

            # 取平均作为类别嵌入
            self.embedder.category_embeddings[category] = embeddings.mean(dim=0)
            print(f"  {category}: {len(templates)} 个模板")

        # 构建类别矩阵
        self.embedder.categories = LABELS
        self.embedder.category_matrix = torch.stack([
            self.embedder.category_embeddings[c] for c in LABELS
        ])
        self.embedder._is_trained = True

    def train_from_data(
        self,
        train_data: List[Dict],
        batch_size: int = 32,
    ) -> Dict[str, int]:
        """
        从标注数据训练，对难样本增加权重

        原理：
        - 对于属于混淆类别对的样本，在计算类别中心时增加权重
        - 例如：时政类的样本如果和军事容易混淆，则该样本权重 x2

        Args:
            train_data: 训练数据
            batch_size: 批处理大小

        Returns:
            每个类别的样本数
        """
        print("\n" + "=" * 60)
        print(" Hard Negative Mining 训练")
        print("=" * 60)

        # 按类别分组
        category_samples = {label: [] for label in LABELS}
        for item in train_data:
            label = item.get('label')
            if label and label in category_samples:
                category_samples[label].append(item)

        # 统计
        sample_counts = {label: len(samples) for label, samples in category_samples.items()}
        print(f"\n训练数据分布:")
        for label, count in sample_counts.items():
            is_confusing = "⚠️" if label in self.confusing_categories else ""
            print(f"  {label}: {count} 条 {is_confusing}")
        print(f"  总计: {sum(sample_counts.values())} 条")

        # 为每个类别计算加权嵌入
        print("\n计算加权类别嵌入...")
        start_time = time.time()

        for label in LABELS:
            samples = category_samples[label]
            if not samples:
                print(f"  警告: {label} 无训练样本，使用增强模板")
                continue

            # 构建输入并计算权重
            all_inputs = []
            all_weights = []

            for item in samples:
                title = item.get('title', '')
                content = item.get('text', item.get('content', ''))

                text_parts = []
                if title:
                    text_parts.append(f"标题：{title}")
                if content:
                    text_parts.append(f"内容：{content[:500]}")

                input_text = "\n".join(text_parts) if text_parts else "无内容"
                all_inputs.append({
                    "text": input_text,
                    "instruction": "表示这篇网页内容的主题类别。"
                })

                # 判断是否为混淆类别的样本
                weight = 1.0
                if label in self.confusing_categories:
                    weight = self.hard_negative_weight
                all_weights.append(weight)

            # 批量计算嵌入
            all_embeddings = []
            for i in range(0, len(all_inputs), batch_size):
                batch_inputs = all_inputs[i:i + batch_size]
                batch_embeddings = self.embedder.embedder.process(batch_inputs)
                all_embeddings.append(batch_embeddings)

            # 合并嵌入
            category_embeddings = torch.cat(all_embeddings, dim=0)

            # 加权平均
            weights_tensor = torch.tensor(all_weights, device=category_embeddings.device)
            weights_tensor = weights_tensor.unsqueeze(1)  # [N, 1]
            weighted_sum = (category_embeddings * weights_tensor).sum(dim=0)
            weighted_mean = weighted_sum / weights_tensor.sum()

            # 与增强模板嵌入融合（保留区分信息）
            template_embedding = self.embedder.category_embeddings[label]
            # 50% 数据 + 50% 模板
            self.embedder.category_embeddings[label] = 0.5 * weighted_mean + 0.5 * template_embedding

            avg_weight = sum(all_weights) / len(all_weights)
            print(f"  {label}: {len(samples)} 样本, 平均权重 {avg_weight:.2f}x")

        # 重建类别矩阵
        self.embedder.category_matrix = torch.stack([
            self.embedder.category_embeddings[c] for c in LABELS
        ])

        train_time = time.time() - start_time
        print(f"\n训练完成! 耗时: {train_time:.1f}s")
        print("=" * 60)

        return sample_counts

    def classify(
        self,
        title: str,
        content: str = "",
        url: str = "",
    ) -> HardNegativeResult:
        """
        分类并检测是否为难样本

        Args:
            title: 标题
            content: 内容
            url: URL

        Returns:
            HardNegativeResult: 分类结果（含难样本分析）
        """
        start_time = time.time()

        # 调用基础分类器
        base_result = self.embedder.classify(title=title, content=content, url=url)

        # 分析是否为难样本
        sorted_scores = sorted(base_result.scores.items(), key=lambda x: -x[1])
        top1_label, top1_score = sorted_scores[0]
        top2_label, top2_score = sorted_scores[1] if len(sorted_scores) > 1 else (None, 0)

        margin = top1_score - top2_score

        # 判断是否为难样本
        is_hard_negative = False
        confusing_pair = None

        if margin < self.margin_threshold and top2_label:
            # margin 很小，检查是否为已知混淆对
            pair = (top1_label, top2_label)
            if pair in self.confusing_pairs_set:
                is_hard_negative = True
                confusing_pair = pair

        latency_ms = (time.time() - start_time) * 1000

        return HardNegativeResult(
            label=base_result.label,
            confidence=base_result.confidence,
            scores=base_result.scores,
            latency_ms=latency_ms,
            is_hard_negative=is_hard_negative,
            confusing_pair=confusing_pair,
            margin=margin,
        )

    def batch_classify(
        self,
        items: List[Dict],
    ) -> List[HardNegativeResult]:
        """批量分类"""
        results = []
        hard_negative_count = 0

        print(f"\n  {'='*60}")
        print(f"  Hard Negative Mining 分类 (共 {len(items)} 条)")
        print(f"  {'='*60}")

        for idx, item in enumerate(items):
            title = item.get("title", "")
            content = item.get("text", item.get("content", ""))
            url = item.get("articleLink", item.get("url", ""))

            result = self.classify(title=title, content=content, url=url)
            results.append(result)

            if result.is_hard_negative:
                hard_negative_count += 1

            # 打印进度
            item_id = item.get("id", idx)
            hn_flag = "⚠️" if result.is_hard_negative else ""
            print(f"  [{item_id:4d}] {title[:35]:35s} → {result.label} {hn_flag}")

            if result.is_hard_negative:
                print(f"         难样本: {result.confusing_pair[0]}↔{result.confusing_pair[1]}, margin={result.margin:.3f}")

            if (idx + 1) % 10 == 0:
                print(f"  --- 进度: {idx + 1}/{len(items)}, 难样本: {hard_negative_count} ---")

        print(f"  {'='*60}")
        print(f"  难样本统计: {hard_negative_count}/{len(items)} ({hard_negative_count/len(items)*100:.1f}%)")
        print(f"  {'='*60}\n")

        return results

    def analyze_confusion_matrix(
        self,
        test_data: List[Dict],
        results: List[HardNegativeResult],
    ) -> Dict:
        """
        分析混淆矩阵，找出需要挖掘的难样本

        Returns:
            混淆矩阵分析结果
        """
        # 构建混淆矩阵
        confusion = defaultdict(lambda: defaultdict(int))

        for item, result in zip(test_data, results):
            true_label = item.get("label")
            pred_label = result.label
            confusion[true_label][pred_label] += 1

        # 分析容易混淆的类别对
        confusing_analysis = []

        for true_label in LABELS:
            for pred_label in LABELS:
                if true_label != pred_label:
                    count = confusion[true_label][pred_label]
                    if count > 0:
                        # 检查是否为已知混淆对
                        is_known_pair = (true_label, pred_label) in self.confusing_pairs_set
                        confusing_analysis.append({
                            "true": true_label,
                            "pred": pred_label,
                            "count": count,
                            "is_known_pair": is_known_pair,
                        })

        # 按混淆次数排序
        confusing_analysis.sort(key=lambda x: -x["count"])

        return {
            "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
            "confusing_pairs": confusing_analysis,
        }

    def save_category_matrix(self, path: str):
        """保存类别嵌入"""
        self.embedder.save_category_matrix(path)

    def load_category_matrix(self, path: str):
        """加载类别嵌入"""
        self.embedder.load_category_matrix(path)


def train_and_evaluate(
    data_path: str,
    train_count: int = 500,
    test_count: int = 100,
    hard_negative_weight: float = 2.0,
    margin_threshold: float = 0.05,
    save_path: str = None,
    load_path: str = None,
):
    """
    训练并评估 Hard Negative Mining 分类器
    """
    print("=" * 70)
    print(" Hard Negative Mining 分类器 - 训练与评估")
    print("=" * 70)
    print(f"数据文件: {data_path}")
    print(f"训练样本: {train_count} 条")
    print(f"测试样本: {test_count} 条")
    print(f"难样本权重: {hard_negative_weight}x")
    print(f"Margin 阈值: {margin_threshold}")
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
    classifier = HardNegativeMiningClassifier(
        hard_negative_weight=hard_negative_weight,
        margin_threshold=margin_threshold,
    )

    # 训练或加载
    if load_path:
        print("\n[3/5] 加载已训练的类别嵌入...")
        classifier.load_category_matrix(load_path)
    else:
        print("\n[3/5] 训练...")
        classifier.train_from_data(train_data)
        if save_path:
            classifier.save_category_matrix(save_path)

    # 测试
    print("\n[4/5] 测试...")
    results = classifier.batch_classify(test_data)

    # 评估
    print("\n[5/5] 评估...")
    correct = 0
    hard_negative_errors = 0
    errors = []

    for item, result in zip(test_data, results):
        true_label = item.get("label")
        pred_label = result.label

        if true_label == pred_label:
            correct += 1
        else:
            errors.append({
                "id": item.get("id"),
                "title": item.get("title", "")[:50],
                "true": true_label,
                "pred": pred_label,
                "is_hard_negative": result.is_hard_negative,
                "margin": result.margin,
            })
            if result.is_hard_negative:
                hard_negative_errors += 1

    accuracy = correct / len(test_data) if test_data else 0

    # 混淆矩阵分析
    confusion_analysis = classifier.analyze_confusion_matrix(test_data, results)

    # 各类别准确率
    per_class_metrics = {}
    for label in LABELS:
        label_items = [(item, result) for item, result in zip(test_data, results)
                       if item.get("label") == label]
        if label_items:
            label_correct = sum(1 for item, result in label_items
                               if item.get("label") == result.label)
            is_confusing = label in classifier.confusing_categories
            per_class_metrics[label] = {
                'accuracy': label_correct / len(label_items),
                'correct': label_correct,
                'total': len(label_items),
                'is_confusing_category': is_confusing,
            }

    # 打印结果
    print("\n" + "=" * 70)
    print(" 评估结果")
    print("=" * 70)
    print(f"\n  准确率: {accuracy:.2%} ({correct}/{len(test_data)})")
    print(f"  难样本错误: {hard_negative_errors}/{len(errors)} ({hard_negative_errors/max(len(errors),1)*100:.1f}%)")

    print("\n  各类别准确率:")
    for label, stats in per_class_metrics.items():
        flag = "⚠️" if stats.get('is_confusing_category') else ""
        print(f"    {label}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']}) {flag}")

    print("\n  主要混淆类别对:")
    for pair in confusion_analysis["confusing_pairs"][:5]:
        known = "✓" if pair["is_known_pair"] else ""
        print(f"    {pair['true']} → {pair['pred']}: {pair['count']} 次 {known}")

    if errors:
        print(f"\n  错误样本 (共 {len(errors)} 条):")
        for err in errors[:5]:
            hn_flag = "⚠️难样本" if err['is_hard_negative'] else ""
            print(f"    [{err['id']:4d}] {err['title'][:40]}")
            print(f"           真实: {err['true']}, 预测: {err['pred']}, margin={err['margin']:.3f} {hn_flag}")

    # 保存结果
    result_dir = Path("/home/zzh/webpage-classification/data/results")
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_filename = f"hard_negative_mining_train{train_count}_test{test_count}_{timestamp}.json"
    result_path = result_dir / result_filename

    result_data = {
        'timestamp': timestamp,
        'config': {
            'method': 'Hard Negative Mining',
            'train_count': train_count,
            'test_count': test_count,
            'hard_negative_weight': hard_negative_weight,
            'margin_threshold': margin_threshold,
        },
        'overall': {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(test_data),
            'hard_negative_errors': hard_negative_errors,
        },
        'per_class': per_class_metrics,
        'confusion_analysis': confusion_analysis,
        'errors': errors,
    }

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"\n  结果已保存至: {result_path}")
    print("=" * 70)

    return accuracy, errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hard Negative Mining 分类器")
    parser.add_argument("--data", default="/home/zzh/webpage-classification/data/labeled/labeled.jsonl",
                       help="标注数据路径")
    parser.add_argument("--train", type=int, default=500, help="训练样本数")
    parser.add_argument("--test", type=int, default=100, help="测试样本数")
    parser.add_argument("--top-k", type=int, default=1,
                       help="Top-K: 1=单阶段Embedding, 2/3/4=两阶段+Reranker (默认: 1)")
    parser.add_argument("--weight", type=float, default=2.0, help="难样本权重倍数")
    parser.add_argument("--margin", type=float, default=0.05, help="难样本判定阈值")
    parser.add_argument("--save", help="保存类别嵌入的路径")
    parser.add_argument("--load", help="加载已训练的类别嵌入")

    args = parser.parse_args()

    # 自动生成保存路径 (Embedding 矩阵与 top_k 无关)
    save_path = args.save or f"/home/zzh/webpage-classification/models/hard_negative_train{args.train}.pt"

    # 自动保存 log
    result_dir = Path("/home/zzh/webpage-classification/data/results")
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.top_k == 1:
        log_path = result_dir / f"hard_negative_mining_train{args.train}_test{args.test}_{timestamp}.log"
    else:
        log_path = result_dir / f"hard_negative_mining_train{args.train}_topk{args.top_k}_test{args.test}_{timestamp}.log"
    logger = TeeLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    print(f"Log 保存至: {log_path}")

    if args.top_k == 1:
        # 单阶段 Embedding 模式
        train_and_evaluate(
            data_path=args.data,
            train_count=args.train,
            test_count=args.test,
            hard_negative_weight=args.weight,
            margin_threshold=args.margin,
            save_path=save_path,
            load_path=args.load,
        )
    else:
        # 两阶段 Embedding + Reranker 模式
        # 先训练并保存嵌入
        if not args.load:
            print("=" * 70)
            print(" Hard Negative Mining 分类器 - 训练阶段")
            print("=" * 70)
            all_data = load_labeled_data(args.data)
            train_data = all_data[:args.train]

            classifier = HardNegativeMiningClassifier(
                hard_negative_weight=args.weight,
                margin_threshold=args.margin,
            )
            classifier.train_from_data(train_data)
            classifier.save_category_matrix(save_path)
            args.load = save_path

        # 调用 two_stage_classifier 进行评估
        from two_stage_classifier import train_and_evaluate as two_stage_eval
        two_stage_eval(
            data_path=args.data,
            train_count=args.train,
            test_count=args.test,
            top_k=args.top_k,
            load_path=args.load,
            doc_version="v2",
            result_prefix="hard_negative",
        )
