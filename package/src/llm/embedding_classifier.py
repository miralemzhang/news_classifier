"""
基于 Qwen3-VL-Embedding 的网页分类器（生产版）

使用嵌入相似度进行分类，支持单条和批量推理。
"""

import torch
import torch.nn.functional as F
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingClassifyResult:
    """分类结果"""
    label: str
    confidence: float
    scores: Dict[str, float]
    latency_ms: float


# 8 类分类体系
LABELS = ["时政", "经济", "军事", "社会", "科技", "体育", "娱乐", "其他"]

# Softmax temperature for confidence scoring
DEFAULT_TEMPERATURE = 10.0

# 各类别的描述模板（用于未训练时的初始化）
CATEGORY_TEMPLATES = {
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


class QwenEmbeddingClassifier:
    """基于 Qwen3-VL-Embedding 的分类器（生产版）"""

    def __init__(
        self,
        model_path: str,
        device: str = None,
        torch_dtype=None,
        use_template: bool = True,
        max_content_length: int = 500,
    ):
        try:
            from .qwen3_vl_embedding import Qwen3VLEmbedder
        except ImportError:
            from qwen3_vl_embedding import Qwen3VLEmbedder

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        logger.info(f"Loading model: {model_path} (device={device}, dtype={torch_dtype})")

        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            torch_dtype=torch_dtype,
        )
        self.device = device
        self.max_content_length = max_content_length

        self.category_embeddings = {}
        self.categories = LABELS
        self.category_matrix = None
        self._is_trained = False

        if use_template:
            self._init_category_embeddings()
        else:
            logger.info("Model loaded, waiting for train_from_data() or load_category_matrix().")

        logger.info("Model ready.")

    def _init_category_embeddings(self):
        """预计算各类别模板的嵌入向量"""
        logger.info("Computing category template embeddings...")

        for category, templates in CATEGORY_TEMPLATES.items():
            inputs = [{"text": t, "instruction": "表示这段文本的类别特征。"} for t in templates]
            embeddings = self.embedder.process(inputs)
            self.category_embeddings[category] = embeddings.mean(dim=0)

        self.categories = list(self.category_embeddings.keys())
        self.category_matrix = torch.stack([
            self.category_embeddings[c] for c in self.categories
        ])
        # category_matrix shape: [num_categories, embed_dim]

        logger.info(f"Initialized {len(self.categories)} category embeddings.")
        self._is_trained = True

    def train_from_data(
        self,
        train_data: List[Dict],
        batch_size: int = 32,
    ) -> Dict[str, int]:
        """
        从标注数据中训练类别嵌入

        Args:
            train_data: 训练数据列表，每项需包含 title, text/content, label
            batch_size: 批处理大小

        Returns:
            每个类别的训练样本数
        """
        logger.info("Training category embeddings from labeled data...")

        category_samples = {label: [] for label in LABELS}
        for item in train_data:
            label = item.get('label')
            if label and label in category_samples:
                category_samples[label].append(item)

        sample_counts = {label: len(samples) for label, samples in category_samples.items()}
        for label, count in sample_counts.items():
            logger.info(f"  {label}: {count} samples")
        logger.info(f"  Total: {sum(sample_counts.values())} samples")

        start_time = time.time()

        for label in LABELS:
            samples = category_samples[label]
            if not samples:
                logger.warning(f"{label}: no training samples, using template fallback")
                templates = CATEGORY_TEMPLATES.get(label, ["无"])
                inputs = [{"text": t, "instruction": "表示这段文本的类别特征。"} for t in templates]
                embeddings = self.embedder.process(inputs)
                self.category_embeddings[label] = embeddings.mean(dim=0)
                continue

            all_inputs = []
            for item in samples:
                title = item.get('title', '')
                content = item.get('text', item.get('content', ''))
                text_parts = []
                if title:
                    text_parts.append(f"标题：{title}")
                if content:
                    text_parts.append(f"内容：{content[:self.max_content_length]}")
                input_text = "\n".join(text_parts) if text_parts else "无内容"
                all_inputs.append({
                    "text": input_text,
                    "instruction": "表示这篇网页内容的主题类别。"
                })

            all_embeddings = []
            for i in range(0, len(all_inputs), batch_size):
                batch_inputs = all_inputs[i:i + batch_size]
                batch_embeddings = self.embedder.process(batch_inputs)
                all_embeddings.append(batch_embeddings)

            category_embeddings = torch.cat(all_embeddings, dim=0)
            self.category_embeddings[label] = category_embeddings.mean(dim=0)
            logger.info(f"  {label}: trained on {len(samples)} samples")

        self.categories = LABELS
        self.category_matrix = torch.stack([
            self.category_embeddings[c] for c in self.categories
        ])

        self._is_trained = True
        logger.info(f"Training completed in {time.time() - start_time:.1f}s")

        return sample_counts

    def save_category_matrix(self, path: str):
        """保存类别嵌入矩阵"""
        if not self._is_trained:
            logger.warning("Model not trained; saving template-initialized embeddings.")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'category_matrix': self.category_matrix,
            'category_embeddings': dict(self.category_embeddings),
            'categories': self.categories,
            'is_trained': self._is_trained,
        }
        torch.save(save_data, path)
        logger.info(f"Category embeddings saved to: {path}")

    def load_category_matrix(self, path: str):
        """加载类别嵌入矩阵"""
        try:
            data = torch.load(path, map_location=self.device, weights_only=False)
        except (FileNotFoundError, RuntimeError) as e:
            raise RuntimeError(f"Failed to load category matrix from {path}: {e}")
        self.category_matrix = data['category_matrix'].to(self.device)
        self.category_embeddings = {k: v.to(self.device) for k, v in data['category_embeddings'].items()}
        self.categories = data['categories']
        self._is_trained = data.get('is_trained', True)
        logger.info(f"Category embeddings loaded: {path} ({len(self.categories)} categories, dim={self.category_matrix.shape[1]})")

    def _build_input_text(self, title: str, url: str = "", content: str = "") -> str:
        """构建分类输入文本"""
        text_parts = []
        if title:
            text_parts.append(f"标题：{title}")
        if url:
            text_parts.append(f"URL：{url}")
        if content:
            text_parts.append(f"内容：{content[:self.max_content_length]}")
        return "\n".join(text_parts) if text_parts else "无内容"

    @torch.no_grad()
    def classify(
        self,
        title: str,
        url: str = "",
        content: str = "",
        candidate_labels: List[str] = None,
    ) -> EmbeddingClassifyResult:
        """
        对单条输入进行分类

        Args:
            title: 标题
            url: URL
            content: 正文内容
            candidate_labels: 候选类别列表（默认全部）

        Returns:
            EmbeddingClassifyResult
        """
        if candidate_labels is None:
            candidate_labels = self.categories

        start_time = time.time()

        input_text = self._build_input_text(title, url, content)
        inputs = [{"text": input_text, "instruction": "表示这篇网页内容的主题类别。"}]
        text_embedding = self.embedder.process(inputs)[0]  # [embed_dim]

        # 构建候选类别的索引和矩阵
        candidate_indices = [i for i, c in enumerate(self.categories) if c in candidate_labels]
        candidate_matrix = self.category_matrix[candidate_indices]  # [K, embed_dim]
        candidate_names = [self.categories[i] for i in candidate_indices]

        # 矢量化余弦相似度: [K]
        similarities = F.cosine_similarity(
            text_embedding.unsqueeze(0),  # [1, embed_dim]
            candidate_matrix,             # [K, embed_dim]
        )

        scores = dict(zip(candidate_names, similarities.tolist()))

        # Softmax 置信度
        confidences = F.softmax(similarities * DEFAULT_TEMPERATURE, dim=0)
        best_idx = similarities.argmax().item()
        best_label = candidate_names[best_idx]
        confidence = confidences[best_idx].item()

        latency_ms = (time.time() - start_time) * 1000

        return EmbeddingClassifyResult(
            label=best_label,
            confidence=confidence,
            scores=scores,
            latency_ms=latency_ms,
        )

    @torch.no_grad()
    def batch_classify(
        self,
        items: List[Dict],
        candidate_labels: List[str] = None,
        batch_size: int = 128,
    ) -> List[EmbeddingClassifyResult]:
        """
        批量分类

        Args:
            items: 输入列表，每项包含 title, url/articleLink, text/content
            candidate_labels: 候选类别列表
            batch_size: 每批嵌入计算的数量

        Returns:
            分类结果列表
        """
        if candidate_labels is None:
            candidate_labels = self.categories

        start_time = time.time()

        # 构建输入文本
        all_inputs = []
        for item in items:
            title = item.get("title", "")
            url = item.get("articleLink", item.get("url", ""))
            content = item.get("text", item.get("content", ""))
            input_text = self._build_input_text(title, url, content)
            all_inputs.append({"text": input_text, "instruction": "表示这篇网页内容的主题类别。"})

        # 分批计算嵌入
        all_embeddings = []
        for i in range(0, len(all_inputs), batch_size):
            batch_inputs = all_inputs[i:i + batch_size]
            batch_embeddings = self.embedder.process(batch_inputs)
            all_embeddings.append(batch_embeddings)
            if len(items) > batch_size:
                processed = min(i + batch_size, len(items))
                logger.info(f"Embedding progress: {processed}/{len(items)}")

        text_embeddings = torch.cat(all_embeddings, dim=0)  # [N, embed_dim]

        embed_time = time.time() - start_time
        logger.info(f"Embeddings computed: {len(items)} items in {embed_time:.1f}s ({embed_time/len(items)*1000:.1f}ms/item)")

        # 构建候选类别
        candidate_indices = [i for i, c in enumerate(self.categories) if c in candidate_labels]
        candidate_matrix = self.category_matrix[candidate_indices]  # [K, embed_dim]
        candidate_names = [self.categories[i] for i in candidate_indices]

        # 矢量化: 一次计算所有样本与所有类别的相似度
        # text_embeddings: [N, D], candidate_matrix: [K, D]
        # 因为向量已经 L2 归一化，cosine similarity = dot product
        similarity_matrix = text_embeddings @ candidate_matrix.T  # [N, K]

        # 批量 softmax
        confidence_matrix = F.softmax(similarity_matrix * DEFAULT_TEMPERATURE, dim=1)  # [N, K]

        total_time = time.time() - start_time
        avg_latency = (total_time / len(items)) * 1000

        # 构建结果
        results = []
        for idx in range(len(items)):
            sims = similarity_matrix[idx]
            confs = confidence_matrix[idx]
            best_idx = sims.argmax().item()

            scores = dict(zip(candidate_names, sims.tolist()))

            results.append(EmbeddingClassifyResult(
                label=candidate_names[best_idx],
                confidence=confs[best_idx].item(),
                scores=scores,
                latency_ms=avg_latency,
            ))

        logger.info(f"Batch classification done: {len(items)} items, {total_time:.1f}s total, {avg_latency:.1f}ms/item avg")

        return results
