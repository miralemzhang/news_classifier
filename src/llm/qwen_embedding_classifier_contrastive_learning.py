"""
基于对比学习 (Contrastive Learning) 微调的网页分类器

原理：
- 用标注数据微调 Embedding 模型
- 拉近同类样本、拉远异类样本
- Loss = -log(exp(sim(A,B)/τ) / Σexp(sim(A,neg)/τ))

训练流程：
1. 冻结 Qwen3-VL-Embedding 大部分层
2. 只训练最后几层 + 投影头
3. 用 InfoNCE Loss 优化

输入格式: (anchor, positive, negative)
例如: (时政_A, 时政_B, 经济_C)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import json
import random
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# 支持直接运行和作为模块导入
try:
    from .qwen3_vl_embedding import Qwen3VLEmbedder, Qwen3VLForEmbedding
    from .embedding_classifier import LABELS, EmbeddingClassifyResult
except ImportError:
    from qwen3_vl_embedding import Qwen3VLEmbedder, Qwen3VLForEmbedding
    from embedding_classifier import LABELS, EmbeddingClassifyResult

logger = logging.getLogger(__name__)


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


# ═══════════════════════════════════════════════════════════════════════════════
# 数据集类
# ═══════════════════════════════════════════════════════════════════════════════

class ContrastiveDataset(Dataset):
    """
    对比学习数据集

    生成三元组: (anchor, positive, negative)
    - anchor: 锚点样本
    - positive: 同类样本
    - negative: 异类样本
    """

    def __init__(
        self,
        data: List[Dict],
        num_negatives: int = 5,
        hard_negative_ratio: float = 0.5,
    ):
        """
        Args:
            data: 标注数据列表，每项需包含 title, text, label
            num_negatives: 每个锚点的负样本数量
            hard_negative_ratio: 困难负样本的比例 (来自相似类别)
        """
        self.data = data
        self.num_negatives = num_negatives
        self.hard_negative_ratio = hard_negative_ratio

        # 按类别分组
        self.samples_by_label = defaultdict(list)
        for idx, item in enumerate(data):
            label = item.get('label')
            if label and label in LABELS:
                self.samples_by_label[label].append(idx)

        # 定义类别相似度（用于困难负样本挖掘）
        # 相似类别对：容易混淆的类别
        self.similar_categories = {
            "时政": ["军事", "社会"],
            "经济": ["科技", "社会"],
            "军事": ["时政", "社会"],
            "社会": ["时政", "经济"],
            "科技": ["经济", "其他"],
            "体育": ["娱乐", "社会"],
            "娱乐": ["体育", "社会"],
            "其他": ["社会", "科技"],
        }

        # 构建有效的锚点索引列表（类别至少有2个样本才能构建正样本对）
        self.valid_anchors = []
        for label, indices in self.samples_by_label.items():
            if len(indices) >= 2:
                self.valid_anchors.extend(indices)

        print(f"对比学习数据集初始化完成:")
        print(f"  总样本数: {len(data)}")
        print(f"  有效锚点数: {len(self.valid_anchors)}")
        for label, indices in self.samples_by_label.items():
            print(f"  {label}: {len(indices)} 条")

    def __len__(self):
        return len(self.valid_anchors)

    def __getitem__(self, idx) -> Dict:
        """
        返回一个训练样本: anchor + positive + negatives
        """
        anchor_idx = self.valid_anchors[idx]
        anchor_item = self.data[anchor_idx]
        anchor_label = anchor_item['label']

        # 获取同类样本作为 positive
        same_class_indices = [i for i in self.samples_by_label[anchor_label] if i != anchor_idx]
        positive_idx = random.choice(same_class_indices)
        positive_item = self.data[positive_idx]

        # 获取异类样本作为 negatives
        negative_indices = []
        all_other_labels = [l for l in LABELS if l != anchor_label]

        # 困难负样本（来自相似类别）
        num_hard = int(self.num_negatives * self.hard_negative_ratio)
        similar_labels = self.similar_categories.get(anchor_label, [])
        for sim_label in similar_labels:
            if sim_label in self.samples_by_label and len(negative_indices) < num_hard:
                candidates = self.samples_by_label[sim_label]
                if candidates:
                    negative_indices.append(random.choice(candidates))

        # 随机负样本（来自其他类别）
        while len(negative_indices) < self.num_negatives:
            rand_label = random.choice(all_other_labels)
            if rand_label in self.samples_by_label:
                candidates = self.samples_by_label[rand_label]
                if candidates:
                    neg_idx = random.choice(candidates)
                    if neg_idx not in negative_indices:
                        negative_indices.append(neg_idx)

        negative_items = [self.data[i] for i in negative_indices]

        return {
            'anchor': self._format_text(anchor_item),
            'positive': self._format_text(positive_item),
            'negatives': [self._format_text(item) for item in negative_items],
            'anchor_label': anchor_label,
        }

    def _format_text(self, item: Dict) -> str:
        """格式化样本文本"""
        title = item.get('title', '')
        content = item.get('text', item.get('content', ''))

        text_parts = []
        if title:
            text_parts.append(f"标题：{title}")
        if content:
            text_parts.append(f"内容：{content[:500]}")

        return "\n".join(text_parts) if text_parts else "无内容"


# ═══════════════════════════════════════════════════════════════════════════════
# 损失函数
# ═══════════════════════════════════════════════════════════════════════════════

class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (NT-Xent Loss 的变体)

    Loss = -log(exp(sim(anchor, pos)/τ) / (exp(sim(anchor, pos)/τ) + Σexp(sim(anchor, neg)/τ)))

    其中:
    - sim: 余弦相似度
    - τ: 温度参数 (temperature)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor_emb: torch.Tensor,      # [batch_size, hidden_dim]
        positive_emb: torch.Tensor,    # [batch_size, hidden_dim]
        negative_embs: torch.Tensor,   # [batch_size, num_negatives, hidden_dim]
    ) -> torch.Tensor:
        """
        计算 InfoNCE Loss

        Args:
            anchor_emb: 锚点嵌入
            positive_emb: 正样本嵌入
            negative_embs: 负样本嵌入

        Returns:
            loss: 标量损失值
        """
        batch_size = anchor_emb.size(0)

        # 计算锚点与正样本的相似度: [batch_size]
        pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=-1)
        pos_sim = pos_sim / self.temperature

        # 计算锚点与所有负样本的相似度: [batch_size, num_negatives]
        # anchor_emb: [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim]
        anchor_expanded = anchor_emb.unsqueeze(1)
        neg_sim = F.cosine_similarity(anchor_expanded, negative_embs, dim=-1)
        neg_sim = neg_sim / self.temperature

        # InfoNCE Loss
        # numerator: exp(sim(anchor, pos)/τ)
        # denominator: exp(sim(anchor, pos)/τ) + Σexp(sim(anchor, neg)/τ)

        # 方式1: 使用 logsumexp 数值稳定
        # [batch_size, 1 + num_negatives]
        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

        # log_softmax 的第一个位置就是正样本
        log_softmax = F.log_softmax(all_sim, dim=1)
        loss = -log_softmax[:, 0].mean()

        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss

    Loss = max(0, margin + d(anchor, positive) - d(anchor, negative))

    其中 d 是距离函数 (这里用 1 - cosine_similarity)
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 Triplet Loss
        """
        # 距离 = 1 - 余弦相似度
        pos_dist = 1 - F.cosine_similarity(anchor_emb, positive_emb, dim=-1)

        # 对每个负样本计算 triplet loss，取平均
        anchor_expanded = anchor_emb.unsqueeze(1)
        neg_dist = 1 - F.cosine_similarity(anchor_expanded, negative_embs, dim=-1)

        # loss = max(0, margin + pos_dist - neg_dist)
        losses = F.relu(self.margin + pos_dist.unsqueeze(1) - neg_dist)
        loss = losses.mean()

        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# 投影头 (Projection Head)
# ═══════════════════════════════════════════════════════════════════════════════

class ProjectionHead(nn.Module):
    """
    投影头：将嵌入投影到对比学习空间

    结构：Linear -> ReLU -> Linear
    """

    def __init__(self, input_dim: int, hidden_dim: int = 1024, output_dim: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 对比学习分类器
# ═══════════════════════════════════════════════════════════════════════════════

class ContrastiveLearningClassifier:
    """
    基于对比学习微调的网页分类器

    训练流程:
    1. 加载预训练的 Qwen3-VL-Embedding 模型
    2. 冻结大部分层，只训练最后几层
    3. 添加投影头
    4. 使用 InfoNCE Loss 进行对比学习
    5. 训练完成后，使用微调后的模型进行分类
    """

    def __init__(
        self,
        model_path: str,
        device: str = None,
        torch_dtype = None,
        num_trainable_layers: int = 2,
        projection_dim: int = 256,
        use_projection_head: bool = True,
    ):
        """
        Args:
            model_path: Qwen3-VL-Embedding 模型路径
            device: 设备
            torch_dtype: 数据类型
            num_trainable_layers: 要训练的最后几层 Transformer 层数
            projection_dim: 投影头输出维度
            use_projection_head: 是否使用投影头
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self.device = device
        self.torch_dtype = torch_dtype
        self.num_trainable_layers = num_trainable_layers
        self.projection_dim = projection_dim
        self.use_projection_head = use_projection_head

        print(f"正在加载模型: {model_path}")
        print(f"设备: {device}, 数据类型: {torch_dtype}")
        print(f"可训练层数: {num_trainable_layers}")

        # 加载 embedder
        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            torch_dtype=torch_dtype,
        )

        # 获取隐藏层维度 (兼容不同模型配置)
        config = self.embedder.model.config
        if hasattr(config, 'hidden_size'):
            self.hidden_dim = config.hidden_size
        elif hasattr(config, 'd_model'):
            self.hidden_dim = config.d_model
        elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            self.hidden_dim = config.text_config.hidden_size
        else:
            # 从模型输出推断
            self.hidden_dim = 4096  # Qwen3-VL-8B 默认
            print(f"警告: 无法从配置获取 hidden_size，使用默认值 {self.hidden_dim}")
        print(f"隐藏层维度: {self.hidden_dim}")

        # 创建投影头
        if use_projection_head:
            self.projection_head = ProjectionHead(
                input_dim=self.hidden_dim,
                hidden_dim=1024,
                output_dim=projection_dim,
            ).to(device)
            print(f"投影头: {self.hidden_dim} -> 1024 -> {projection_dim}")
        else:
            self.projection_head = None

        # 类别嵌入（训练后计算）
        self.category_embeddings = {}
        self.category_matrix = None
        self.categories = LABELS
        self._is_trained = False

        print("模型加载完成!")

    def _freeze_layers(self):
        """
        冻结模型的大部分层，只保留最后几层可训练
        """
        model = self.embedder.model

        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False

        # 尝试找到 Transformer 层 (兼容不同模型结构)
        layers = None
        norm = None

        # 尝试多种可能的路径
        possible_paths = [
            # Qwen3-VL-Embedding 可能的结构
            lambda m: m.model.language_model.model.layers if hasattr(m, 'model') and hasattr(m.model, 'language_model') and hasattr(m.model.language_model, 'model') else None,
            lambda m: m.model.language_model.layers if hasattr(m, 'model') and hasattr(m.model, 'language_model') and hasattr(m.model.language_model, 'layers') else None,
            lambda m: m.language_model.model.layers if hasattr(m, 'language_model') and hasattr(m.language_model, 'model') else None,
            lambda m: m.language_model.layers if hasattr(m, 'language_model') and hasattr(m.language_model, 'layers') else None,
            lambda m: m.model.layers if hasattr(m, 'model') and hasattr(m.model, 'layers') else None,
            lambda m: m.layers if hasattr(m, 'layers') else None,
        ]

        for path_fn in possible_paths:
            try:
                layers = path_fn(model)
                if layers is not None:
                    break
            except:
                continue

        if layers is not None:
            total_layers = len(layers)
            print(f"总 Transformer 层数: {total_layers}")
            print(f"解冻最后 {self.num_trainable_layers} 层")

            for i in range(total_layers - self.num_trainable_layers, total_layers):
                for param in layers[i].parameters():
                    param.requires_grad = True
                print(f"  解冻层 {i}")

            # 尝试找到并解冻 norm 层
            norm_paths = [
                lambda m: m.model.language_model.model.norm if hasattr(m, 'model') and hasattr(m.model, 'language_model') and hasattr(m.model.language_model, 'model') and hasattr(m.model.language_model.model, 'norm') else None,
                lambda m: m.model.language_model.norm if hasattr(m, 'model') and hasattr(m.model, 'language_model') and hasattr(m.model.language_model, 'norm') else None,
                lambda m: m.language_model.model.norm if hasattr(m, 'language_model') and hasattr(m.language_model, 'model') and hasattr(m.language_model.model, 'norm') else None,
                lambda m: m.language_model.norm if hasattr(m, 'language_model') and hasattr(m.language_model, 'norm') else None,
                lambda m: m.model.norm if hasattr(m, 'model') and hasattr(m.model, 'norm') else None,
                lambda m: m.norm if hasattr(m, 'norm') else None,
            ]

            for norm_fn in norm_paths:
                try:
                    norm = norm_fn(model)
                    if norm is not None:
                        for param in norm.parameters():
                            param.requires_grad = True
                        print("  解冻 final norm")
                        break
                except:
                    continue
        else:
            print("警告: 无法找到语言模型层，将使用默认冻结策略")

        # 统计可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    def _get_embeddings(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """
        获取文本的嵌入向量

        Args:
            texts: 文本列表
            normalize: 是否归一化

        Returns:
            embeddings: [batch_size, hidden_dim] 或 [batch_size, projection_dim]
        """
        inputs = [{"text": t, "instruction": "表示这篇网页内容的主题类别。"} for t in texts]
        embeddings = self.embedder.process(inputs, normalize=False)

        # 如果使用投影头，则投影到对比学习空间
        if self.projection_head is not None and self.training:
            embeddings = self.projection_head(embeddings.float())

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    @property
    def training(self):
        """检查是否处于训练模式"""
        return self.embedder.model.training

    def train_mode(self):
        """切换到训练模式"""
        self.embedder.model.train()
        if self.projection_head:
            self.projection_head.train()

    def eval_mode(self):
        """切换到评估模式"""
        self.embedder.model.eval()
        if self.projection_head:
            self.projection_head.eval()

    def train(
        self,
        train_data: List[Dict],
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        temperature: float = 0.07,
        num_negatives: int = 5,
        loss_type: str = "infonce",
        accumulation_steps: int = 4,
        save_path: str = None,
        log_interval: int = 10,
    ) -> Dict:
        """
        使用对比学习训练模型

        Args:
            train_data: 训练数据
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            temperature: InfoNCE 温度参数
            num_negatives: 每个样本的负样本数
            loss_type: 损失函数类型 ("infonce" 或 "triplet")
            accumulation_steps: 梯度累积步数
            save_path: 模型保存路径
            log_interval: 日志打印间隔

        Returns:
            训练历史
        """
        print("\n" + "=" * 70)
        print(" 对比学习训练")
        print("=" * 70)
        print(f"训练配置:")
        print(f"  训练样本: {len(train_data)} 条")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Temperature: {temperature}")
        print(f"  负样本数: {num_negatives}")
        print(f"  损失函数: {loss_type}")
        print(f"  梯度累积: {accumulation_steps}")
        print("=" * 70)

        # 冻结层
        self._freeze_layers()

        # 创建数据集和数据加载器
        dataset = ContrastiveDataset(
            data=train_data,
            num_negatives=num_negatives,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self._collate_fn,
        )

        # 创建损失函数
        if loss_type == "infonce":
            criterion = InfoNCELoss(temperature=temperature)
        elif loss_type == "triplet":
            criterion = TripletLoss(margin=0.5)
        else:
            raise ValueError(f"未知的损失函数类型: {loss_type}")

        # 创建优化器
        params_to_optimize = []

        # 添加模型可训练参数
        for param in self.embedder.model.parameters():
            if param.requires_grad:
                params_to_optimize.append(param)

        # 添加投影头参数
        if self.projection_head:
            params_to_optimize.extend(self.projection_head.parameters())

        optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)

        # 学习率调度器
        total_steps = len(dataloader) * epochs // accumulation_steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        # 训练循环
        self.train_mode()
        history = {'loss': [], 'lr': []}
        global_step = 0

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch_idx, batch in enumerate(progress_bar):
                # 获取嵌入
                anchor_emb = self._get_embeddings(batch['anchor'])
                positive_emb = self._get_embeddings(batch['positive'])

                # 处理负样本
                # batch['negatives'] 是 [batch_size, num_negatives] 的文本列表
                negative_texts = batch['negatives']
                batch_size_actual = len(batch['anchor'])

                # 将负样本展平，统一计算嵌入
                flat_negatives = []
                for neg_list in negative_texts:
                    flat_negatives.extend(neg_list)

                if flat_negatives:
                    neg_embs = self._get_embeddings(flat_negatives)
                    # 重塑为 [batch_size, num_negatives, hidden_dim]
                    neg_embs = neg_embs.view(batch_size_actual, num_negatives, -1)
                else:
                    continue

                # 计算损失
                loss = criterion(anchor_emb, positive_emb, neg_embs)
                loss = loss / accumulation_steps
                loss.backward()

                epoch_loss += loss.item() * accumulation_steps
                num_batches += 1

                # 梯度累积
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * accumulation_steps:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                })

                # 记录历史
                if (batch_idx + 1) % log_interval == 0:
                    history['loss'].append(loss.item() * accumulation_steps)
                    history['lr'].append(scheduler.get_last_lr()[0])

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch+1}/{epochs} - 平均损失: {avg_loss:.4f}")

        # 切换到评估模式
        self.eval_mode()

        # 使用训练数据计算类别嵌入
        print("\n计算类别嵌入...")
        self._compute_category_embeddings(train_data)

        # 保存模型
        if save_path:
            self.save(save_path)

        self._is_trained = True
        print("\n训练完成!")

        return history

    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """
        数据批处理函数
        """
        anchors = [item['anchor'] for item in batch]
        positives = [item['positive'] for item in batch]
        negatives = [item['negatives'] for item in batch]
        labels = [item['anchor_label'] for item in batch]

        return {
            'anchor': anchors,
            'positive': positives,
            'negatives': negatives,
            'labels': labels,
        }

    def _compute_category_embeddings(self, train_data: List[Dict], batch_size: int = 32):
        """
        使用训练数据计算类别嵌入
        """
        # 按类别分组
        samples_by_label = defaultdict(list)
        for item in train_data:
            label = item.get('label')
            if label and label in LABELS:
                samples_by_label[label].append(item)

        self.eval_mode()

        with torch.no_grad():
            for label in LABELS:
                samples = samples_by_label.get(label, [])

                if not samples:
                    print(f"  警告: {label} 无训练样本，使用模板初始化")
                    doc = RERANKER_DOCS_V2.get(label, "无")
                    templates = [t.strip() for t in doc.split("；") if t.strip()][:3]
                    if not templates:
                        templates = ["无"]
                    embeddings = self._get_embeddings(templates)
                    self.category_embeddings[label] = embeddings.mean(dim=0)
                    continue

                # 构建文本
                texts = []
                for item in samples:
                    title = item.get('title', '')
                    content = item.get('text', item.get('content', ''))
                    text_parts = []
                    if title:
                        text_parts.append(f"标题：{title}")
                    if content:
                        text_parts.append(f"内容：{content[:500]}")
                    texts.append("\n".join(text_parts) if text_parts else "无内容")

                # 批量计算嵌入
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embs = self._get_embeddings(batch_texts)
                    all_embeddings.append(batch_embs)

                category_embs = torch.cat(all_embeddings, dim=0)
                self.category_embeddings[label] = category_embs.mean(dim=0)
                print(f"  {label}: 使用 {len(samples)} 个样本")

        # 构建类别矩阵
        self.categories = LABELS
        self.category_matrix = torch.stack([
            self.category_embeddings[c] for c in self.categories
        ])

    def classify(
        self,
        title: str,
        url: str = "",
        content: str = "",
        candidate_labels: List[str] = None,
    ) -> EmbeddingClassifyResult:
        """
        对输入文本进行分类
        """
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

        # 计算嵌入
        self.eval_mode()
        with torch.no_grad():
            text_embedding = self._get_embeddings([input_text])[0]

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
        batch_size: int = 32,
    ) -> List[EmbeddingClassifyResult]:
        """
        批量分类
        """
        if candidate_labels is None:
            candidate_labels = self.categories

        start_time = time.time()

        # 构建所有输入文本
        all_texts = []
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

            all_texts.append("\n".join(text_parts) if text_parts else "无内容")

        # 批量计算嵌入
        self.eval_mode()
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i:i + batch_size]
                batch_embs = self._get_embeddings(batch_texts)
                all_embeddings.append(batch_embs)

                if len(items) > batch_size:
                    processed = min(i + batch_size, len(items))
                    print(f"  嵌入进度: {processed}/{len(items)}")

        text_embeddings = torch.cat(all_embeddings, dim=0)

        total_time = time.time() - start_time
        avg_latency = (total_time / len(items)) * 1000

        # 计算分类结果
        results = []
        print(f"\n  {'='*60}")
        print(f"  对比学习分类结果 (共 {len(text_embeddings)} 条)")
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

            # 输出日志
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

    def save(self, path: str):
        """
        保存模型

        Args:
            path: 保存路径
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'category_matrix': self.category_matrix,
            'category_embeddings': {k: v for k, v in self.category_embeddings.items()},
            'categories': self.categories,
            'projection_head_state': self.projection_head.state_dict() if self.projection_head else None,
            'hidden_dim': self.hidden_dim,
            'projection_dim': self.projection_dim,
            'use_projection_head': self.use_projection_head,
            'is_trained': self._is_trained,
        }
        torch.save(save_data, path)
        print(f"模型已保存至: {path}")

    def load(self, path: str):
        """
        加载模型

        Args:
            path: 模型路径
        """
        data = torch.load(path, map_location=self.device)

        self.category_matrix = data['category_matrix'].to(self.device)
        self.category_embeddings = {k: v.to(self.device) for k, v in data['category_embeddings'].items()}
        self.categories = data['categories']
        self._is_trained = data.get('is_trained', True)

        # 加载投影头
        if data.get('projection_head_state') and self.projection_head:
            self.projection_head.load_state_dict(data['projection_head_state'])

        print(f"模型已加载: {path}")
        print(f"  类别数: {len(self.categories)}")
        print(f"  向量维度: {self.category_matrix.shape[1]}")


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def load_labeled_data(file_path: str) -> List[Dict]:
    """加载标注数据"""
    import re
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
    test_count: int = 150,
    model_path: str = None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    num_trainable_layers: int = 2,
    save_path: str = None,
):
    """
    训练并评估对比学习分类器
    """
    print("=" * 70)
    print(" 对比学习分类器 - 训练与评估")
    print("=" * 70)
    print(f"数据文件: {data_path}")
    print(f"训练样本: {train_count} 条")
    print(f"测试样本: {test_count} 条")
    print(f"训练轮数: {epochs}")
    print(f"可训练层数: {num_trainable_layers}")
    print("=" * 70)

    # 加载数据
    print("\n[1/5] 加载数据...")
    all_data = load_labeled_data(data_path)
    print(f"  总计: {len(all_data)} 条")

    train_data = all_data[:train_count]
    test_data = all_data[train_count:train_count + test_count]
    print(f"  训练集: {len(train_data)} 条")
    print(f"  测试集: {len(test_data)} 条")

    # 创建分类器
    print("\n[2/5] 加载模型...")
    if model_path is None:
        model_path = "/home/zzh/webpage-classification/models/Qwen3-VL-Embedding-8B"

    classifier = ContrastiveLearningClassifier(
        model_path=model_path,
        num_trainable_layers=num_trainable_layers,
    )

    # 训练
    print("\n[3/5] 对比学习训练...")
    history = classifier.train(
        train_data=train_data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_path=save_path,
    )

    # 测试
    print("\n[4/5] 测试...")
    results = classifier.batch_classify(test_data)

    # 评估
    print("\n[5/5] 评估...")
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
                'confidence': result.confidence,
            })

    accuracy = correct / len(test_data) if test_data else 0

    # 打印结果
    print("\n" + "=" * 70)
    print(" 评估结果")
    print("=" * 70)
    print(f"\n准确率: {accuracy:.2%} ({correct}/{len(test_data)})")

    # 按类别统计
    print("\n各类别准确率:")
    per_class_metrics = {}
    for label in LABELS:
        label_items = [(item, result) for item, result in zip(test_data, results)
                       if item.get('label') == label]
        if label_items:
            label_correct = sum(1 for item, result in label_items
                               if item.get('label') == result.label)
            label_acc = label_correct / len(label_items)
            per_class_metrics[label] = {
                'accuracy': label_acc,
                'correct': label_correct,
                'total': len(label_items),
            }
            print(f"  {label}: {label_acc:.2%} ({label_correct}/{len(label_items)})")

    # 显示错误样本
    if errors:
        print(f"\n错误样本 (共 {len(errors)} 条):")
        for err in errors[:10]:
            print(f"  [{err['id']:4d}] {err['title'][:40]:40s}")
            print(f"         真实: {err['true']}, 预测: {err['pred']} ({err['confidence']:.2f})")

    # 保存结果
    result_dir = Path("data/results")
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_filename = f"contrastive_train{train_count}_test{test_count}_{timestamp}.json"
    result_path = result_dir / result_filename

    result_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'train_count': train_count,
            'test_count': test_count,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_trainable_layers': num_trainable_layers,
        },
        'overall': {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(test_data),
        },
        'per_class': per_class_metrics,
        'errors': errors,
        'history': {
            'loss': [float(l) for l in history['loss'][-20:]],  # 保存最后20个loss值
        }
    }

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至: {result_path}")
    print("=" * 70)

    return accuracy, errors


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="对比学习网页分类器")
    parser.add_argument("--data", default="/home/zzh/webpage-classification/data/labeled/labeled.jsonl",
                       help="标注数据路径")
    parser.add_argument("--train", type=int, default=500, help="训练样本数")
    parser.add_argument("--test", type=int, default=150, help="测试样本数")
    parser.add_argument("--top-k", type=int, default=1,
                       help="Top-K: 1=单阶段Embedding, 2/3/4=两阶段+Reranker (默认: 1)")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--trainable-layers", type=int, default=2, help="可训练层数")
    parser.add_argument("--model", help="模型路径")
    parser.add_argument("--save", help="保存路径")
    parser.add_argument("--load", help="加载已训练模型")

    args = parser.parse_args()

    # 自动生成保存路径 (Embedding 矩阵与 top_k 无关)
    save_path = args.save or f"/home/zzh/webpage-classification/models/contrastive_train{args.train}_layers{args.trainable_layers}.pt"

    # 自动保存 log
    result_dir = Path("/home/zzh/webpage-classification/data/results")
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.top_k == 1:
        log_path = result_dir / f"contrastive_train{args.train}_test{args.test}_{timestamp}.log"
    else:
        log_path = result_dir / f"contrastive_train{args.train}_topk{args.top_k}_test{args.test}_{timestamp}.log"
    logger = TeeLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    print(f"Log 保存至: {log_path}")

    if args.top_k == 1:
        # 单阶段 Embedding 模式
        if args.load:
            # 加载模式：直接测试
            print("=" * 70)
            print(" 对比学习分类器 - 加载模式")
            print("=" * 70)

            all_data = load_labeled_data(args.data)
            test_data = all_data[args.train:args.train + args.test]

            model_path = args.model or "/home/zzh/webpage-classification/models/Qwen3-VL-Embedding-8B"
            classifier = ContrastiveLearningClassifier(model_path=model_path)
            classifier.load(args.load)

            results = classifier.batch_classify(test_data)

            correct = sum(1 for item, result in zip(test_data, results)
                         if item.get('label') == result.label)
            accuracy = correct / len(test_data) if test_data else 0
            print(f"\n准确率: {accuracy:.2%} ({correct}/{len(test_data)})")
        else:
            # 训练+测试模式
            train_and_evaluate(
                data_path=args.data,
                train_count=args.train,
                test_count=args.test,
                model_path=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                num_trainable_layers=args.trainable_layers,
                save_path=save_path,
            )
    else:
        # 两阶段 Embedding + Reranker 模式
        # 先训练对比学习分类器并保存嵌入
        if not args.load:
            print("=" * 70)
            print(" 对比学习分类器 - 训练阶段")
            print("=" * 70)
            model_path = args.model or "/home/zzh/webpage-classification/models/Qwen3-VL-Embedding-8B"
            all_data = load_labeled_data(args.data)
            train_data = all_data[:args.train]

            classifier = ContrastiveLearningClassifier(
                model_path=model_path,
                num_trainable_layers=args.trainable_layers,
            )
            classifier.train(
                train_data=train_data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                save_path=save_path,
            )
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
            result_prefix="contrastive",
        )
