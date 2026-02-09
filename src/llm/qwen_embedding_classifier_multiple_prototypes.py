"""
多原型嵌入分类器 (Multiple Prototypes Embedding Classifier)

原理：每个类别用 K 个向量表示，捕捉类内多样性

当前:  时政 → 1 个向量 (所有样本平均)
改为:  时政 → K 个向量 (K-Means 聚类中心)

┌─────────────────────────────────────────────────────────────┐
│ 训练阶段                                                     │
│                                                             │
│  时政样本 (240 条)                                           │
│      │                                                      │
│      ▼                                                      │
│  计算所有嵌入 → K-Means 聚类 (K=3)                            │
│      │                                                      │
│      ▼                                                      │
│  3 个聚类中心:                                               │
│    - 时政_1: 外交新闻 [4096]                                 │
│    - 时政_2: 国内政策 [4096]                                 │
│    - 时政_3: 领导人活动 [4096]                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 推理阶段                                                     │
│                                                             │
│  新文章 → 嵌入 [4096]                                        │
│      │                                                      │
│      ▼                                                      │
│  与所有原型计算相似度 (8类 × K原型 = 8K 个)                   │
│      │                                                      │
│      ▼                                                      │
│  每类取最大相似度作为类别得分                                 │
│  score(时政) = max(sim(时政_1), sim(时政_2), sim(时政_3))     │
└─────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn.functional as F
import time
import json
import re
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
from sklearn.cluster import KMeans
import numpy as np


@dataclass
class MultiPrototypeClassifyResult:
    """多原型分类结果"""
    label: str                          # 分类标签
    confidence: float                   # 置信度
    scores: Dict[str, float]            # 各类别得分 (取 max)
    prototype_scores: Dict[str, List[float]]  # 各类别各原型得分
    latency_ms: float                   # 耗时（毫秒）


# 8 类分类体系
LABELS = ["时政", "经济", "军事", "社会", "科技", "体育", "娱乐", "其他"]


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


# 类别描述文档 V2 (用于样本不足时的后备)
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


class QwenMultiPrototypeClassifier:
    """基于多原型的 Qwen3-VL-Embedding 分类器"""

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        torch_dtype=None,
        n_prototypes: int = 3,
    ):
        """
        初始化多原型分类器

        Args:
            model_path: 模型路径
            device: 设备 (cuda/cpu)
            torch_dtype: 数据类型
            n_prototypes: 每个类别的原型数量 (默认 3)
        """
        # 支持直接运行和作为模块导入
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
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"正在加载模型: {model_path}")
        print(f"设备: {device}, 数据类型: {torch_dtype}")

        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            torch_dtype=torch_dtype,
        )
        self.device = device
        self.n_prototypes = n_prototypes

        # 多原型嵌入: {类别: [原型1, 原型2, ...]}
        self.category_prototypes: Dict[str, torch.Tensor] = {}
        self.categories = LABELS
        self._is_trained = False

        print(f"模型加载完成! 原型数量: {n_prototypes}")

    def train_from_data(
        self,
        train_data: List[Dict],
        batch_size: int = 32,
        min_samples_for_clustering: int = None,
    ) -> Dict[str, int]:
        """
        从标注数据训练多原型类别嵌入

        使用 K-Means 聚类找到每个类别的 K 个原型

        Args:
            train_data: 训练数据列表，每项需包含:
                - title: 标题
                - text/content: 内容
                - label: 真实标签
            batch_size: 批处理大小
            min_samples_for_clustering: 最少需要多少样本才使用聚类
                                        默认为 n_prototypes * 2

        Returns:
            Dict[str, int]: 每个类别的训练样本数
        """
        if min_samples_for_clustering is None:
            min_samples_for_clustering = self.n_prototypes * 2

        print("\n" + "=" * 60)
        print(f" 多原型训练 (K={self.n_prototypes})")
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
            status = "✓" if count >= min_samples_for_clustering else f"(< {min_samples_for_clustering}, 用模板)"
            print(f"  {label}: {count} 条 {status}")
        print(f"  总计: {sum(sample_counts.values())} 条")

        # 为每个类别训练多原型
        print(f"\n计算类别原型 (每类 {self.n_prototypes} 个)...")
        start_time = time.time()

        for label in LABELS:
            samples = category_samples[label]

            if len(samples) < min_samples_for_clustering:
                # 样本不足，使用模板
                print(f"  {label}: 样本不足，使用模板初始化")
                prototypes = self._init_prototypes_from_templates(label)
            else:
                # 样本充足，使用 K-Means 聚类
                prototypes = self._train_prototypes_kmeans(label, samples, batch_size)

            self.category_prototypes[label] = prototypes
            print(f"  {label}: {len(samples)} 样本 → {prototypes.shape[0]} 原型")

        self._is_trained = True
        train_time = time.time() - start_time
        print(f"\n训练完成! 耗时: {train_time:.1f}s")
        print(f"总原型数: {len(LABELS)} 类 × {self.n_prototypes} = {len(LABELS) * self.n_prototypes}")
        print("=" * 60)

        return sample_counts

    def _init_prototypes_from_templates(self, label: str) -> torch.Tensor:
        """使用模板初始化原型 (样本不足时的后备方案)"""
        # 从 RERANKER_DOCS_V2 获取描述，按分号分割成多个子描述
        doc = RERANKER_DOCS_V2.get(label, "无")
        templates = [t.strip() for t in doc.split("；") if t.strip()][:self.n_prototypes]

        if not templates:
            templates = ["无"]

        inputs = [{"text": t, "instruction": "表示这段文本的类别特征。"} for t in templates]
        embeddings = self.embedder.process(inputs)

        # 如果模板数量不足 n_prototypes，复制填充
        if embeddings.shape[0] < self.n_prototypes:
            repeats = (self.n_prototypes // embeddings.shape[0]) + 1
            embeddings = embeddings.repeat(repeats, 1)[:self.n_prototypes]

        return embeddings

    def _train_prototypes_kmeans(
        self,
        label: str,
        samples: List[Dict],
        batch_size: int,
    ) -> torch.Tensor:
        """
        使用 K-Means 聚类训练类别原型

        Args:
            label: 类别名称
            samples: 该类别的所有样本
            batch_size: 批处理大小

        Returns:
            torch.Tensor: 形状为 [n_prototypes, embedding_dim]
        """
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

        # 合并所有嵌入
        embeddings = torch.cat(all_embeddings, dim=0)  # [N, 4096]

        # K-Means 聚类
        embeddings_np = embeddings.cpu().numpy()
        kmeans = KMeans(
            n_clusters=self.n_prototypes,
            random_state=42,
            n_init=10,
        )
        kmeans.fit(embeddings_np)

        # 返回聚类中心作为原型
        prototypes = torch.tensor(
            kmeans.cluster_centers_,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )

        return prototypes

    def save_prototypes(self, path: str):
        """
        保存训练好的多原型嵌入

        Args:
            path: 保存路径
        """
        if not self._is_trained:
            print("警告: 模型未训练")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'category_prototypes': {k: v.cpu() for k, v in self.category_prototypes.items()},
            'categories': self.categories,
            'n_prototypes': self.n_prototypes,
            'is_trained': self._is_trained,
        }
        torch.save(save_data, path)
        print(f"多原型嵌入已保存至: {path}")

    def save_as_single_embedding(self, path: str):
        """
        保存为单嵌入格式（兼容 two_stage_classifier）
        每个类别取多原型的平均作为单一嵌入

        Args:
            path: 保存路径
        """
        if not self._is_trained:
            print("警告: 模型未训练")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 将多原型平均为单嵌入
        category_embeddings = {}
        for label, prototypes in self.category_prototypes.items():
            category_embeddings[label] = prototypes.mean(dim=0).cpu()

        category_matrix = torch.stack([category_embeddings[c] for c in self.categories])

        save_data = {
            'category_matrix': category_matrix,
            'category_embeddings': category_embeddings,
            'categories': self.categories,
            'is_trained': self._is_trained,
            'method': f'multi_prototype_k{self.n_prototypes}_avg',
        }
        torch.save(save_data, path)
        print(f"单嵌入格式已保存至: {path} (多原型平均)")

    def load_prototypes(self, path: str):
        """
        加载训练好的多原型嵌入

        Args:
            path: 模型路径
        """
        data = torch.load(path, map_location=self.device)
        self.category_prototypes = {k: v.to(self.device) for k, v in data['category_prototypes'].items()}
        self.categories = data['categories']
        self.n_prototypes = data['n_prototypes']
        self._is_trained = data.get('is_trained', True)

        print(f"多原型嵌入已加载: {path}")
        print(f"  类别数: {len(self.categories)}")
        print(f"  每类原型数: {self.n_prototypes}")
        total_prototypes = sum(v.shape[0] for v in self.category_prototypes.values())
        print(f"  总原型数: {total_prototypes}")

    def classify(
        self,
        title: str,
        url: str = "",
        content: str = "",
    ) -> MultiPrototypeClassifyResult:
        """
        对输入文本进行分类

        使用多原型匹配：每个类别取与其所有原型相似度的最大值

        Args:
            title: 标题
            url: URL
            content: 正文内容

        Returns:
            MultiPrototypeClassifyResult: 分类结果
        """
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
        text_embedding = self.embedder.process(inputs)[0]  # [4096]

        # 计算与各类别各原型的相似度
        scores = {}
        prototype_scores = {}

        for label in self.categories:
            prototypes = self.category_prototypes[label]  # [K, 4096]

            # 计算与该类别所有原型的相似度
            similarities = F.cosine_similarity(
                text_embedding.unsqueeze(0),  # [1, 4096]
                prototypes,                    # [K, 4096]
            )  # [K]

            # 记录各原型得分
            prototype_scores[label] = similarities.tolist()

            # 取最大值作为类别得分
            scores[label] = similarities.max().item()

        # 找到最高得分的类别
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        # 计算置信度 (softmax 归一化)
        score_values = torch.tensor(list(scores.values()))
        confidences = F.softmax(score_values * 10, dim=0)
        confidence = confidences[list(scores.keys()).index(best_label)].item()

        latency_ms = (time.time() - start_time) * 1000

        return MultiPrototypeClassifyResult(
            label=best_label,
            confidence=confidence,
            scores=scores,
            prototype_scores=prototype_scores,
            latency_ms=latency_ms,
        )

    def batch_classify(
        self,
        items: List[Dict],
        batch_size: int = 128,
    ) -> List[MultiPrototypeClassifyResult]:
        """
        批量分类

        Args:
            items: 输入列表
            batch_size: 每批处理的数量

        Returns:
            List[MultiPrototypeClassifyResult]: 分类结果列表
        """
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

        text_embeddings = torch.cat(all_embeddings, dim=0)  # [N, 4096]

        total_time = time.time() - start_time
        avg_latency = (total_time / len(items)) * 1000

        # 计算每个输入的分类结果
        results = []
        print(f"\n  {'='*60}")
        print(f"  多原型分类 (K={self.n_prototypes}) - 共 {len(text_embeddings)} 条")
        print(f"  {'='*60}")

        for idx, embedding in enumerate(text_embeddings):
            scores = {}
            prototype_scores = {}

            for label in self.categories:
                prototypes = self.category_prototypes[label]

                # 计算与该类别所有原型的相似度
                similarities = F.cosine_similarity(
                    embedding.unsqueeze(0),
                    prototypes,
                )

                prototype_scores[label] = similarities.tolist()
                scores[label] = similarities.max().item()

            best_label = max(scores, key=scores.get)

            score_values = torch.tensor(list(scores.values()))
            confidences = F.softmax(score_values * 10, dim=0)
            confidence = confidences[list(scores.keys()).index(best_label)].item()

            # 输出日志
            item = items[idx]
            item_id = item.get('id', idx)
            title = item.get('title', '')[:30]

            scores_str = " | ".join([f"{k}:{v:.3f}" for k, v in sorted(scores.items(), key=lambda x: -x[1])[:4]])
            print(f"  [{item_id:4d}] {title:30s} → {best_label} ({confidence:.2f})")
            print(f"         相似度: {scores_str}")

            results.append(MultiPrototypeClassifyResult(
                label=best_label,
                confidence=confidence,
                scores=scores,
                prototype_scores=prototype_scores,
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
    n_prototypes: int = 3,
    save_path: str = None,
    load_path: str = None,
):
    """
    训练并评估多原型分类器

    Args:
        data_path: 数据路径
        train_count: 训练样本数
        test_count: 测试样本数
        n_prototypes: 每类原型数量
        save_path: 保存路径
        load_path: 加载路径
    """
    print("=" * 70)
    print(f" 多原型嵌入分类器 (K={n_prototypes})")
    print("=" * 70)
    print(f"数据文件: {data_path}")
    print(f"训练样本: {train_count} 条")
    print(f"测试样本: {test_count} 条")
    print(f"原型数量: 每类 {n_prototypes} 个")
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
    classifier = QwenMultiPrototypeClassifier(n_prototypes=n_prototypes)

    # 训练或加载
    if load_path:
        print("\n[3/4] 加载已训练的原型...")
        classifier.load_prototypes(load_path)
    else:
        print("\n[3/4] 训练原型...")
        classifier.train_from_data(train_data)
        if save_path:
            classifier.save_prototypes(save_path)

    # 测试
    print("\n[4/4] 测试...")
    results = classifier.batch_classify(test_data)

    # 评估
    correct = 0
    errors = []
    for item, result in zip(test_data, results):
        true_label = item.get('label')
        pred_label = result.label

        if true_label == pred_label:
            correct += 1
        else:
            errors.append({
                'id': item.get('id'),
                'title': item.get('title', '')[:50],
                'true': true_label,
                'pred': pred_label,
                'scores': result.scores,
            })

    accuracy = correct / len(test_data) if test_data else 0

    # 各类别准确率
    per_class_metrics = {}
    for label in LABELS:
        label_items = [(item, result) for item, result in zip(test_data, results)
                       if item.get('label') == label]
        if label_items:
            label_correct = sum(1 for item, result in label_items
                               if item.get('label') == result.label)
            per_class_metrics[label] = {
                'accuracy': label_correct / len(label_items),
                'correct': label_correct,
                'total': len(label_items),
            }

    # 打印结果
    print("\n" + "=" * 70)
    print(" 评估结果")
    print("=" * 70)
    print(f"\n准确率: {accuracy:.2%} ({correct}/{len(test_data)})")

    print("\n各类别准确率:")
    for label, stats in per_class_metrics.items():
        print(f"  {label}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    if errors:
        print(f"\n错误样本 (共 {len(errors)} 条，显示前 10 条):")
        for err in errors[:10]:
            print(f"  [{err['id']:4d}] {err['title'][:40]}")
            print(f"         真实: {err['true']}, 预测: {err['pred']}")

    # 保存结果
    result_dir = Path("/home/zzh/webpage-classification/data/results")
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_filename = f"multi_prototype_k{n_prototypes}_train{train_count}_test{test_count}_{timestamp}.json"
    result_path = result_dir / result_filename

    result_data = {
        'timestamp': timestamp,
        'config': {
            'method': f'Multi-Prototype (K={n_prototypes})',
            'train_count': train_count,
            'test_count': test_count,
            'n_prototypes': n_prototypes,
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

    print(f"\n结果已保存至: {result_path}")
    print("=" * 70)

    return accuracy, errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="多原型嵌入分类器")
    parser.add_argument("--data", default="/home/zzh/webpage-classification/data/labeled/labeled.jsonl",
                       help="标注数据路径")
    parser.add_argument("--train", type=int, default=500, help="训练样本数")
    parser.add_argument("--test", type=int, default=100, help="测试样本数")
    parser.add_argument("--top-k", type=int, default=1,
                       help="Top-K: 1=单阶段Embedding, 2/3/4=两阶段+Reranker (默认: 1)")
    parser.add_argument("-k", "--n-prototypes", type=int, default=3,
                       help="每个类别的原型数量 (默认: 3)")
    parser.add_argument("--save", help="保存原型的路径")
    parser.add_argument("--load", help="加载已训练的原型")

    args = parser.parse_args()

    # 自动生成保存路径 (Embedding 矩阵与 top_k 无关)
    save_path = args.save or f"/home/zzh/webpage-classification/models/multi_prototype_k{args.n_prototypes}_train{args.train}.pt"

    # 自动保存 log
    result_dir = Path("/home/zzh/webpage-classification/data/results")
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.top_k == 1:
        log_path = result_dir / f"multi_prototype_k{args.n_prototypes}_train{args.train}_test{args.test}_{timestamp}.log"
    else:
        log_path = result_dir / f"multi_prototype_k{args.n_prototypes}_train{args.train}_topk{args.top_k}_test{args.test}_{timestamp}.log"
    logger = TeeLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    print(f"Log 保存至: {log_path}")

    if args.top_k == 1:
        # 单阶段多原型 Embedding 模式
        train_and_evaluate(
            data_path=args.data,
            train_count=args.train,
            test_count=args.test,
            n_prototypes=args.n_prototypes,
            save_path=save_path,
            load_path=args.load,
        )
    else:
        # 两阶段 Embedding + Reranker 模式
        # 先训练多原型分类器并保存为单嵌入格式
        if not args.load:
            print("=" * 70)
            print(f" 多原型分类器 (K={args.n_prototypes}) - 训练阶段")
            print("=" * 70)
            all_data = load_labeled_data(args.data)
            train_data = all_data[:args.train]

            classifier = QwenMultiPrototypeClassifier(n_prototypes=args.n_prototypes)
            classifier.train_from_data(train_data)
            # 保存为单嵌入格式以便 two_stage_classifier 使用
            classifier.save_as_single_embedding(save_path)
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
            result_prefix=f"multi_prototype_k{args.n_prototypes}",
        )
