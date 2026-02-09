
"""
基于 Qwen3-VL-Embedding 的网页分类器 - 加权平均版本

加权平均原理：典型样本（离类别中心近）权重更高

┌─────────────────────────────────────────────────────────────┐
│ 步骤 1: 计算初始类别中心（简单平均）                          │
│                                                             │
│  时政_初始 = mean(所有时政样本)                              │
│                                                             │
│ 步骤 2: 计算每个样本与类别中心的距离                          │
│                                                             │
│  样本1: 距离=0.1 → 权重=0.9 (很典型)                         │
│  样本2: 距离=0.5 → 权重=0.5 (一般)                           │
│  样本3: 距离=0.9 → 权重=0.1 (边缘样本)                        │
│                                                             │
│ 步骤 3: 加权平均                                             │
│                                                             │
│  时政_最终 = Σ(weight_i × embedding_i) / Σ(weight_i)        │
└─────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn.functional as F
import time
import json
import re
import sys
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class EmbeddingClassifyResult:
    """分类结果"""
    label: str
    confidence: float
    scores: Dict[str, float]
    latency_ms: float


# 8 类分类体系
LABELS = ["时政", "经济", "军事", "社会", "科技", "体育", "娱乐", "其他"]


# 类别描述文档 V2
RERANKER_DOCS_V2 = {
    "时政": "国家领导人会议讲话，政府工作报告；两会新闻，党建工作，政府机构改革；选举投票，议会立法；外交访问，国际关系声明；政府预算案讨论；党和国家的重大方针政策；政府会议、文件、公报；习近平等党和国家领导人，新时代新思想；中国特色社会主义",
    "经济": "股票市场行情，基金理财，银行利率；企业财报，上市公司，投资并购；房价走势，物价指数，GDP增长；贸易进出口数据；消费零售，电商平台；汇率、交易、证券、财经",
    "军事": "军队演习，武器装备，国防建设；战争冲突，军事行动，武装冲突；军舰航母，战斗机，导弹；空袭轰炸，炮击袭击；军事部署，边境武装对峙",
    "社会": "民生新闻，社会热点，公共事件，社会保障、社会福利、人民利益；教育医疗，交通出行，社会保障；列车事故，车祸，空难；犯罪案件，社会治安；人道主义危机，饥荒",
    "科技": "人工智能（AI），互联网科技，手机数码，信息技术；科学研究，科技奖项，技术创新，航天航空；新能源汽车，芯片半导体；软件应用，云计算；机器人，无人机技术",
    "体育": "足球篮球比赛，运动员，体育赛事；奥运会，世界杯，NBA；健身运动，马拉松；网球、高尔夫、滑雪；体育明星，球队转会",
    "娱乐": "明星八卦，影视综艺，音乐会、演唱会、音乐剧、表演节目；电影电视剧，选秀节目；游戏动漫，时尚潮流；娱乐圈，粉丝文化；艺术展览，文艺演出；综艺、真人秀",
    "其他": "生活百科，美食旅游；天气预报，星座运势；日常杂谈，情感故事；宠物动物，家居装修；广告、营销、软文，营销推广",
}


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


class QwenEmbeddingClassifierWeightedAvg:
    """基于 Qwen3-VL-Embedding 的分类器 - 加权平均版本"""

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        torch_dtype=None,
    ):
        """
        初始化分类器

        Args:
            model_path: 模型路径
            device: 设备 (cuda/cpu)
            torch_dtype: 数据类型
        """
        try:
            from .qwen3_vl_embedding import Qwen3VLEmbedder
        except ImportError:
            from qwen3_vl_embedding import Qwen3VLEmbedder

        if model_path is None:
            base_dir = Path(__file__).parent.parent.parent
            model_path = str(base_dir / "models" / "Qwen3-VL-Embedding-8B")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        print(f"正在加载模型: {model_path}")
        print(f"设备: {device}, 数据类型: {torch_dtype}")

        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            torch_dtype=torch_dtype,
        )
        self.device = device

        # 类别嵌入
        self.category_embeddings = {}
        self.categories = LABELS
        self.category_matrix = None
        self._is_trained = False

        print("模型加载完成! 等待调用 train_from_data() 训练...")

    def train_from_data(
        self,
        train_data: List[Dict],
        batch_size: int = 32,
        n_iterations: int = 1,
    ) -> Dict[str, int]:
        """
        从标注数据中训练类别嵌入 - 加权平均版本

        原理：
        1. 先计算简单平均作为初始类别中心
        2. 计算每个样本与中心的距离，距离近的样本权重高
        3. 使用加权平均得到最终类别中心

        权重计算: weight = 1 / (1 + distance)
        距离使用: distance = 1 - cosine_similarity

        Args:
            train_data: 训练数据列表
            batch_size: 批处理大小
            n_iterations: 迭代次数（可多次迭代优化中心）

        Returns:
            Dict[str, int]: 每个类别的训练样本数
        """
        print("\n" + "=" * 60)
        print(" 加权平均训练类别嵌入")
        print("=" * 60)

        # 按类别分组
        category_samples = {label: [] for label in LABELS}
        for item in train_data:
            label = item.get('label')
            if label and label in category_samples:
                category_samples[label].append(item)

        # 统计
        sample_counts = {label: len(samples) for label, samples in category_samples.items()}
        print(f"训练数据分布:")
        for label, count in sample_counts.items():
            print(f"  {label}: {count} 条")
        print(f"  总计: {sum(sample_counts.values())} 条")

        # 为每个类别计算嵌入
        print("\n计算样本嵌入...")
        start_time = time.time()

        category_all_embeddings = {}  # 存储每个类别所有样本的嵌入

        for label in LABELS:
            samples = category_samples[label]
            if not samples:
                print(f"  警告: {label} 无训练样本，使用模板初始化")
                doc = RERANKER_DOCS_V2.get(label, "无")
                templates = [t.strip() for t in doc.split("；") if t.strip()][:3]
                if not templates:
                    templates = ["无"]
                inputs = [{"text": t, "instruction": "表示这段文本的类别特征。"} for t in templates]
                embeddings = self.embedder.process(inputs)
                self.category_embeddings[label] = embeddings.mean(dim=0)
                category_all_embeddings[label] = None  # 标记为模板初始化
                continue

            # 构建输入
            all_inputs = []
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

            # 批量计算嵌入
            all_embeddings = []
            for i in range(0, len(all_inputs), batch_size):
                batch_inputs = all_inputs[i:i + batch_size]
                batch_embeddings = self.embedder.process(batch_inputs)
                all_embeddings.append(batch_embeddings)

            # 合并
            embeddings = torch.cat(all_embeddings, dim=0)
            category_all_embeddings[label] = embeddings
            print(f"  {label}: {len(samples)} 个样本的嵌入已计算")

        # 加权平均计算
        print(f"\n加权平均计算 (迭代 {n_iterations} 次)...")

        for iteration in range(n_iterations):
            print(f"\n  --- 迭代 {iteration + 1}/{n_iterations} ---")

            for label in LABELS:
                embeddings = category_all_embeddings[label]
                if embeddings is None:
                    continue  # 跳过模板初始化的类别

                # 步骤 1: 计算初始中心（简单平均，或上一轮的中心）
                if iteration == 0:
                    center = embeddings.mean(dim=0)
                else:
                    center = self.category_embeddings[label]

                # 步骤 2: 计算每个样本与中心的距离和权重
                # cosine_similarity 返回 [n_samples]
                similarities = F.cosine_similarity(embeddings, center.unsqueeze(0))
                distances = 1 - similarities  # 距离 = 1 - 相似度
                weights = 1 / (1 + distances)  # 权重 = 1 / (1 + 距离)

                # 步骤 3: 加权平均
                # embeddings: [n_samples, dim]
                # weights: [n_samples]
                weighted_sum = (embeddings * weights.unsqueeze(1)).sum(dim=0)
                weighted_center = weighted_sum / weights.sum()

                # 归一化
                weighted_center = F.normalize(weighted_center, p=2, dim=0)

                self.category_embeddings[label] = weighted_center

                # 打印统计
                min_w, max_w, mean_w = weights.min().item(), weights.max().item(), weights.mean().item()
                print(f"    {label}: 权重范围 [{min_w:.3f}, {max_w:.3f}], 平均 {mean_w:.3f}")

        # 构建类别矩阵
        self.categories = LABELS
        self.category_matrix = torch.stack([
            self.category_embeddings[c] for c in self.categories
        ])

        self._is_trained = True
        train_time = time.time() - start_time
        print(f"\n训练完成! 耗时: {train_time:.1f}s")
        print("=" * 60)

        return sample_counts

    def save_category_matrix(self, path: str):
        """保存训练好的类别嵌入矩阵"""
        if not self._is_trained:
            print("警告: 模型未训练")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'category_matrix': self.category_matrix,
            'category_embeddings': {k: v for k, v in self.category_embeddings.items()},
            'categories': self.categories,
            'is_trained': self._is_trained,
            'method': 'weighted_avg',
        }
        torch.save(save_data, path)
        print(f"类别嵌入已保存至: {path}")

    def load_category_matrix(self, path: str):
        """加载训练好的类别嵌入矩阵"""
        data = torch.load(path, map_location=self.device)
        self.category_matrix = data['category_matrix'].to(self.device)
        self.category_embeddings = {k: v.to(self.device) for k, v in data['category_embeddings'].items()}
        self.categories = data['categories']
        self._is_trained = data.get('is_trained', True)
        method = data.get('method', 'unknown')
        print(f"类别嵌入已加载: {path}")
        print(f"  类别数: {len(self.categories)}, 方法: {method}")

    def classify(
        self,
        title: str,
        url: str = "",
        content: str = "",
        candidate_labels: List[str] = None,
    ) -> EmbeddingClassifyResult:
        """对输入文本进行分类"""
        if candidate_labels is None:
            candidate_labels = self.categories

        start_time = time.time()

        # 构建输入文本
        text_parts = []
        if title:
            text_parts.append(f"标题：{title}")
        if url:
            text_parts.append(f"URL：{url}")
        if content:
            text_parts.append(f"内容：{content[:500]}")

        input_text = "\n".join(text_parts) if text_parts else "无内容"

        # 计算输入文本的嵌入
        inputs = [{"text": input_text, "instruction": "表示这篇网页内容的主题类别。"}]
        text_embedding = self.embedder.process(inputs)[0]

        # 计算与各类别的相似度
        scores = {}
        for i, category in enumerate(self.categories):
            if category in candidate_labels:
                similarity = F.cosine_similarity(
                    text_embedding.unsqueeze(0),
                    self.category_matrix[i].unsqueeze(0)
                ).item()
                scores[category] = similarity

        # 找到最高得分的类别
        best_label = max(scores, key=scores.get)

        # 计算置信度
        score_values = torch.tensor(list(scores.values()))
        confidences = F.softmax(score_values * 10, dim=0)
        confidence = confidences[list(scores.keys()).index(best_label)].item()

        latency_ms = (time.time() - start_time) * 1000

        return EmbeddingClassifyResult(
            label=best_label,
            confidence=confidence,
            scores=scores,
            latency_ms=latency_ms,
        )

    def batch_classify(
        self,
        items: List[Dict],
        candidate_labels: List[str] = None,
        batch_size: int = 128,
    ) -> List[EmbeddingClassifyResult]:
        """批量分类"""
        if candidate_labels is None:
            candidate_labels = self.categories

        start_time = time.time()

        # 构建所有输入文本
        all_inputs = []
        for item in items:
            title = item.get("title", "")
            url = item.get("articleLink", item.get("url", ""))
            content = item.get("text", item.get("content", ""))

            text_parts = []
            if title:
                text_parts.append(f"标题：{title}")
            if url:
                text_parts.append(f"URL：{url}")
            if content:
                text_parts.append(f"内容：{content[:500]}")

            input_text = "\n".join(text_parts) if text_parts else "无内容"
            all_inputs.append({"text": input_text, "instruction": "表示这篇网页内容的主题类别。"})

        # 分批计算嵌入
        all_embeddings = []
        for i in range(0, len(all_inputs), batch_size):
            batch_inputs = all_inputs[i:i + batch_size]
            batch_embeddings = self.embedder.process(batch_inputs)
            all_embeddings.append(batch_embeddings)

            if len(items) > batch_size:
                processed = min(i + batch_size, len(items))
                print(f"  嵌入进度: {processed}/{len(items)}")

        text_embeddings = torch.cat(all_embeddings, dim=0)

        total_time = time.time() - start_time
        avg_latency = (total_time / len(items)) * 1000

        # 计算每个输入的分类结果
        results = []
        print(f"\n  {'='*60}")
        print(f"  分类结果 (共 {len(text_embeddings)} 条)")
        print(f"  {'='*60}")

        for idx, embedding in enumerate(text_embeddings):
            scores = {}
            for i, category in enumerate(self.categories):
                if category in candidate_labels:
                    similarity = F.cosine_similarity(
                        embedding.unsqueeze(0),
                        self.category_matrix[i].unsqueeze(0)
                    ).item()
                    scores[category] = similarity

            best_label = max(scores, key=scores.get)

            score_values = torch.tensor(list(scores.values()))
            confidences = F.softmax(score_values * 10, dim=0)
            confidence = confidences[list(scores.keys()).index(best_label)].item()

            # 输出
            item = items[idx]
            item_id = item.get('id', idx)
            title = item.get('title', '')[:30]

            scores_str = " | ".join([f"{k}:{v:.3f}" for k, v in sorted(scores.items(), key=lambda x: -x[1])[:4]])
            print(f"  [{item_id:4d}] {title:30s} → {best_label} ({confidence:.2f})")
            print(f"         相似度: {scores_str}")

            results.append(EmbeddingClassifyResult(
                label=best_label,
                confidence=confidence,
                scores=scores,
                latency_ms=avg_latency,
            ))

        print(f"  {'='*60}\n")
        return results


def load_labeled_data(file_path: str) -> List[Dict]:
    """加载标注数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    first_char = content.strip()[0] if content.strip() else ''

    if first_char == '[':
        return json.loads(content)
    elif first_char == '{':
        pattern = r'\{\s*"id":\s*\d+.*?\n\}'
        matches = re.findall(pattern, content, re.DOTALL)
        objects = []
        for m in matches:
            m = m.rstrip().rstrip(',')
            try:
                obj = json.loads(m)
                objects.append(obj)
            except json.JSONDecodeError:
                pass
        return objects
    else:
        return [json.loads(line) for line in content.split('\n') if line.strip()]


def train_and_evaluate(
    data_path: str,
    train_count: int = 500,
    test_count: int = 100,
    n_iterations: int = 1,
    save_path: str = None,
    load_path: str = None,
):
    """训练并评估加权平均分类器"""
    print("=" * 70)
    print(" 加权平均 Embedding 分类器 - 训练与评估")
    print("=" * 70)
    print(f"数据文件: {data_path}")
    print(f"训练样本: {train_count} 条")
    print(f"测试样本: {test_count} 条")
    print(f"加权迭代: {n_iterations} 次")
    print("=" * 70)

    # 加载数据
    print("\n[1/4] 加载数据...")
    all_data = load_labeled_data(data_path)
    print(f"  总计: {len(all_data)} 条")

    train_data = all_data[:train_count]
    test_data = all_data[train_count:train_count + test_count]
    print(f"  训练集: {len(train_data)} 条")
    print(f"  测试集: {len(test_data)} 条")

    # 初始化分类器
    print("\n[2/4] 初始化分类器...")
    classifier = QwenEmbeddingClassifierWeightedAvg()

    # 训练或加载
    if load_path:
        print("\n[3/4] 加载已训练的类别嵌入...")
        classifier.load_category_matrix(load_path)
    else:
        print("\n[3/4] 训练...")
        classifier.train_from_data(train_data, n_iterations=n_iterations)
        if save_path:
            classifier.save_category_matrix(save_path)

    # 测试
    print("\n[4/4] 测试...")
    results = classifier.batch_classify(test_data)

    # 评估
    correct = 0
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
                "confidence": result.confidence,
            })

    accuracy = correct / len(test_data) if test_data else 0

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
    print(f"\n  准确率: {accuracy:.2%} ({correct}/{len(test_data)})")

    print("\n  各类别准确率:")
    for label, stats in per_class_metrics.items():
        print(f"    {label}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    if errors:
        print(f"\n  错误样本 (共 {len(errors)} 条，显示前 10 条):")
        for err in errors[:10]:
            print(f"    [{err['id']:4d}] {err['title'][:40]}")
            print(f"           真实: {err['true']}, 预测: {err['pred']} ({err['confidence']:.2f})")

    # 保存结果
    result_dir = Path("/home/zzh/webpage-classification/data/results")
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_filename = f"weighted_avg_train{train_count}_test{test_count}_{timestamp}.json"
    result_path = result_dir / result_filename

    result_data = {
        'timestamp': timestamp,
        'config': {
            'method': 'weighted_avg',
            'train_count': train_count,
            'test_count': test_count,
            'n_iterations': n_iterations,
        },
        'overall': {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(test_data),
        },
        'per_class': per_class_metrics,
        'errors': errors,
    }

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"\n  结果已保存至: {result_path}")
    print("=" * 70)

    return accuracy, errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="加权平均 Embedding 分类器")
    parser.add_argument("--data", default="/home/zzh/webpage-classification/data/labeled/labeled.jsonl",
                       help="标注数据路径")
    parser.add_argument("--train", type=int, default=500, help="训练样本数")
    parser.add_argument("--test", type=int, default=100, help="测试样本数")
    parser.add_argument("--top-k", type=int, default=1,
                       help="Top-K: 1=单阶段Embedding, 2/3/4=两阶段+Reranker (默认: 1)")
    parser.add_argument("--iterations", type=int, default=1, help="加权平均迭代次数")
    parser.add_argument("--save", help="保存类别嵌入的路径")
    parser.add_argument("--load", help="加载已训练的类别嵌入")

    args = parser.parse_args()

    # 自动生成保存路径 (Embedding 矩阵与 top_k 无关)
    save_path = args.save or f"/home/zzh/webpage-classification/models/weighted_avg_train{args.train}.pt"

    # 自动保存 log
    result_dir = Path("/home/zzh/webpage-classification/data/results")
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.top_k == 1:
        log_path = result_dir / f"weighted_avg_train{args.train}_test{args.test}_{timestamp}.log"
    else:
        log_path = result_dir / f"weighted_avg_train{args.train}_topk{args.top_k}_test{args.test}_{timestamp}.log"
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
            n_iterations=args.iterations,
            save_path=save_path,
            load_path=args.load,
        )
    else:
        # 两阶段 Embedding + Reranker 模式
        # 先训练加权平均分类器并保存嵌入
        if not args.load:
            print("=" * 70)
            print(" 加权平均分类器 - 训练阶段")
            print("=" * 70)
            all_data = load_labeled_data(args.data)
            train_data = all_data[:args.train]

            classifier = QwenEmbeddingClassifierWeightedAvg()
            classifier.train_from_data(train_data, n_iterations=args.iterations)
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
            result_prefix="weighted_avg",
        )
