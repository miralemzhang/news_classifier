"""
基于 Qwen3-VL-Embedding 的网页分类器
使用嵌入相似度进行分类

支持两种模式：
1. 模板模式（默认）：使用预定义的类别描述模板
2. 训练模式：从标注数据中学习类别嵌入
"""

import torch
import torch.nn.functional as F
import time
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass


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
class EmbeddingClassifyResult:
    """分类结果"""
    label: str  # 分类标签，如 "经济"
    confidence: float # 置信度，如 0.35
    scores: Dict[str, float]  # 各类别得分
    latency_ms: float # 耗时（毫秒）


# 8 类分类体系
LABELS = ["时政", "经济", "军事", "社会", "科技", "体育", "娱乐", "其他"]  


# 各类别的描述模板
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

'''
    图解：                                                                                    
    CATEGORY_TEMPLATES 的作用                                                                 
    ═══════════════════════════════════════════════════════════════                           
                                                                                            
    "经济" 类别有 3 个描述：                                                                  
    ┌─────────────────────────────────────────────────────────────┐                           
    │ 1. "股票市场行情，基金理财，银行利率调整"                      │                        
    │ 2. "企业财报，上市公司，投资并购，经济数据"                    │                        
    │ 3. "房价走势，物价指数，GDP增长，贸易进出口"                   │                        
    └─────────────────────────────────────────────────────────────┘                           
                                │                                                           
                                ▼                                                           
                            Qwen3 模型                                                        
                                │                                                           
                                ▼                                                           
    ┌─────────────────────────────────────────────────────────────┐                           
    │ 向量1: [0.12, 0.34, 0.56, ...]  (4096维)                     │                          
    │ 向量2: [0.15, 0.32, 0.58, ...]                               │                          
    │ 向量3: [0.11, 0.35, 0.54, ...]                               │                          
    └─────────────────────────────────────────────────────────────┘                           
                                │                                                           
                            取平均                                                           
                                │                                                           
                                ▼                                                           
    ┌─────────────────────────────────────────────────────────────┐                           
    │ V_经济 = [0.127, 0.337, 0.56, ...]                           │                          
    │                                                              │                          
    │ 这个向量代表了"经济"这个概念在向量空间中的位置               │                          
    └─────────────────────────────────────────────────────────────┘   
'''


class QwenEmbeddingClassifier:
    """基于 Qwen3-VL-Embedding 的分类器"""

    def __init__(
        self,
        model_path: str,
        device: str = None,
        torch_dtype = None,
        use_template: bool = True,
    ):
        """
        初始化分类器

        Args:
            model_path: 模型路径
            device: 设备 (cuda/cpu)
            torch_dtype: 数据类型
            use_template: 是否使用预定义模板初始化类别嵌入
                          设为 False 时需要调用 train_from_data() 训练
        """
        # 支持直接运行和作为模块导入
        try:
            from .qwen3_vl_embedding import Qwen3VLEmbedder
        except ImportError:
            from qwen3_vl_embedding import Qwen3VLEmbedder

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

        # 根据模式初始化类别嵌入
        if use_template:
            self._init_category_embeddings()
        else:
            print("模型已加载，等待调用 train_from_data() 训练...")

        print("模型加载完成!")

    def _init_category_embeddings(self):
        """预计算各类别模板的嵌入向量"""
        print("正在计算类别模板嵌入...")

        for category, templates in CATEGORY_TEMPLATES.items():
            # 为每个类别的所有模板计算嵌入
            
            # 例如 category = "经济"                                                          
            # templates = ["股票市场行情...", "企业财报...", "房价走势..."]                                                                                                   
            # 构建输入：每个模板 + 指令                
            inputs = [{"text": t, "instruction": "表示这段文本的类别特征。"} for t in templates]
            
            # 调用模型，得到每个模板的向量                                                                
            embeddings = self.embedder.process(inputs)
            # embeddings 形状: [3, 4096]  (3个模板，每个4096维)                               
            
            # 取平均作为类别嵌入
            self.category_embeddings[category] = embeddings.mean(dim=0)
            # 形状: [4096]

        # 将类别嵌入堆叠成矩阵
        self.categories = list(self.category_embeddings.keys())
        self.category_matrix = torch.stack([
            self.category_embeddings[c] for c in self.categories
        ])
        # category_matrix 形状: [8, 4096]  (8个类别)
        #
        # 图解：初始化后的数据结构
        # self.category_matrix (8 × 4096 矩阵)
        # 每一行代表一个类别的嵌入向量
        # 时政: [0.11, 0.23, 0.45, ...]  ← 4096 个数字
        # 经济: [0.12, 0.34, 0.56, ...]
        # ...
        # 这个矩阵在初始化时就算好了，之后每次分类都可以直接用！

        print(f"已初始化 {len(self.categories)} 个类别的嵌入")
        self._is_trained = True

    def train_from_data(
        self,
        train_data: List[Dict],
        batch_size: int = 32,
    ) -> Dict[str, int]:
        """
        从标注数据中训练类别嵌入

        原理：用每个类别的真实样本文本计算嵌入，取平均作为该类别的代表向量

        Args:
            train_data: 训练数据列表，每项需包含:
                - title: 标题
                - text/content: 内容
                - label: 真实标签
            batch_size: 批处理大小

        Returns:
            Dict[str, int]: 每个类别的训练样本数
        """
        print("\n" + "=" * 60)
        print(" 从标注数据训练类别嵌入")
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
        print("\n计算类别嵌入...")
        start_time = time.time()

        for label in LABELS:
            samples = category_samples[label]
            if not samples:
                print(f"  警告: {label} 无训练样本，使用模板初始化")
                # 使用模板作为后备
                templates = CATEGORY_TEMPLATES.get(label, ["无"])
                inputs = [{"text": t, "instruction": "表示这段文本的类别特征。"} for t in templates]
                embeddings = self.embedder.process(inputs)
                self.category_embeddings[label] = embeddings.mean(dim=0)
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

            # 合并并取平均
            category_embeddings = torch.cat(all_embeddings, dim=0)
            self.category_embeddings[label] = category_embeddings.mean(dim=0)
            print(f"  {label}: 使用 {len(samples)} 个样本")

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
        """
        保存训练好的类别嵌入矩阵

        Args:
            path: 保存路径，如 "models/category_matrix.pt"
        """
        if not self._is_trained:
            print("警告: 模型未训练，保存的是模板初始化的嵌入")

        # 确保目录存在
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'category_matrix': self.category_matrix,
            'category_embeddings': {k: v for k, v in self.category_embeddings.items()},
            'categories': self.categories,
            'is_trained': self._is_trained,
        }
        torch.save(save_data, path)
        print(f"类别嵌入已保存至: {path}")

    def load_category_matrix(self, path: str):
        """
        加载训练好的类别嵌入矩阵

        Args:
            path: 模型路径，如 "category_matrix.pt"
        """
        data = torch.load(path, map_location=self.device)
        self.category_matrix = data['category_matrix'].to(self.device)
        self.category_embeddings = {k: v.to(self.device) for k, v in data['category_embeddings'].items()}
        self.categories = data['categories']
        self._is_trained = data.get('is_trained', True)
        print(f"类别嵌入已加载: {path}")
        print(f"  类别数: {len(self.categories)}")
        print(f"  向量维度: {self.category_matrix.shape[1]}")

    def classify(
        self,
        title: str,
        url: str = "",
        content: str = "",
        candidate_labels: List[str] = None,
    ) -> EmbeddingClassifyResult:
        """
        对输入文本进行分类

        Args:
            title: 标题
            url: URL
            content: 正文内容
            candidate_labels: 候选类别列表

        Returns:
            EmbeddingClassifyResult: 分类结果
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
        # 例如: "标题：央行宣布降息\nURL：https://...\n内容：..."                             

        # 计算输入文本的嵌入
        inputs = [{"text": input_text, "instruction": "表示这篇网页内容的主题类别。"}]
        text_embedding = self.embedder.process(inputs)[0]
        # text_embedding 形状: [4096]                                                         

        # 计算与各类别的相似度
        scores = {}
        for i, category in enumerate(self.categories):
            # 计算余弦相似度
            if category in candidate_labels:
                similarity = F.cosine_similarity(
                    text_embedding.unsqueeze(0), # [1, 4096]
                    self.category_matrix[i].unsqueeze(0) # [1, 4096]
                ).item()
                scores[category] = similarity
                
                # scores 例如: {"时政": 0.62, "经济": 0.89, "军事": 0.31, ...}                        

        # 找到最高得分的类别
        best_label = max(scores, key=scores.get) # 经济
        best_score = scores[best_label]

        # 将相似度转换为置信度 (0-1)
        # 使用 softmax 归一化
        score_values = torch.tensor(list(scores.values()))
        confidences = F.softmax(score_values * 10, dim=0)  # 乘以温度系数
        # softmax 会把分数转换成概率（加起来=1）                                              
        # 乘以 10 是温度系数，让差距更明显
        
        confidence = confidences[list(scores.keys()).index(best_label)].item()
        # 取出"经济"对应的置信度
        
        
        latency_ms = (time.time() - start_time) * 1000

        # 返回结果
        return EmbeddingClassifyResult(
            label=best_label,
            confidence=confidence,
            scores=scores,
            latency_ms=latency_ms,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # 完整流程说明：
    # 1. 初始化时：将 CATEGORY_TEMPLATES 中的描述通过 Qwen3 转成向量矩阵
    # 2. 分类时：输入文本 → Qwen3 → 向量 → 与类别向量计算余弦相似度 → 选最大值
    # ═══════════════════════════════════════════════════════════════════════════

    def batch_classify(
        self,
        items: List[Dict],
        candidate_labels: List[str] = None,
        batch_size: int = 128,
    ) -> List[EmbeddingClassifyResult]:
        """
        批量分类

        Args:
            items: 输入列表，每项包含 title, url, content
            candidate_labels: 候选类别列表
            batch_size: 每批处理的数量（默认8，根据显存调整）

        Returns:
            List[EmbeddingClassifyResult]: 分类结果列表
        """
        if candidate_labels is None:
            candidate_labels = self.categories

        start_time = time.time()

        # 构建所有输入文本（兼容不同字段名）
        all_inputs = []
        for item in items:
            # 兼容新旧字段名
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

            # 显示进度
            if len(items) > batch_size:
                processed = min(i + batch_size, len(items))
                print(f"  嵌入进度: {processed}/{len(items)}")

        # 合并所有批次的嵌入
        text_embeddings = torch.cat(all_embeddings, dim=0)

        total_time = time.time() - start_time
        avg_latency = (total_time / len(items)) * 1000

        # 计算每个输入的分类结果
        results = []
        print(f"\n  {'='*60}")
        print(f"  实时分类日志 (共 {len(text_embeddings)} 条)")
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
            best_score = scores[best_label]

            score_values = torch.tensor(list(scores.values()))
            confidences = F.softmax(score_values * 10, dim=0)  # 与单条分类保持一致
            confidence = confidences[list(scores.keys()).index(best_label)].item()

            # 实时输出相似度矩阵
            item = items[idx]
            item_id = item.get('id', idx)
            title = item.get('title', '')[:30]

            # 格式化分数输出
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
    """
    加载标注数据（支持多行 JSON 格式）

    Args:
        file_path: 文件路径

    Returns:
        List[Dict]: 数据列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    first_char = content.strip()[0] if content.strip() else ''

    if first_char == '[':
        # JSON 数组格式
        return json.loads(content)
    elif first_char == '{':
        # 多行 JSON 对象格式 - 使用正则匹配
        # 匹配 {"id": xxx ... } 格式的对象
        pattern = r'\{\s*"id":\s*\d+.*?\n\}'
        matches = re.findall(pattern, content, re.DOTALL)

        objects = []
        for m in matches:
            # 去掉尾部逗号
            m = m.rstrip().rstrip(',')
            try:
                obj = json.loads(m)
                objects.append(obj)
            except json.JSONDecodeError:
                pass
        return objects
    else:
        # 标准 JSONL 格式
        return [json.loads(line) for line in content.split('\n') if line.strip()]


def create_classifier(
    model_path: str = None,
    device: str = None,
    use_template: bool = True,
) -> QwenEmbeddingClassifier:
    """
    创建分类器的工厂函数

    Args:
        model_path: 模型路径，默认使用配置文件中的路径
        device: 设备 ("cuda" 或 "cpu")，默认自动选择
        use_template: 是否使用预定义模板初始化类别嵌入

    Returns:
        QwenEmbeddingClassifier: 分类器实例
    """
    if model_path is None:
        model_path = "/home/zzh/webpage-classification/models/Qwen3-VL-Embedding-8B"

    return QwenEmbeddingClassifier(
        model_path=model_path,
        device=device,
        use_template=use_template,
    )


def train_and_evaluate(
    data_path: str,
    train_count: int = 200,
    test_count: int = 200,
    model_path: str = None,
    save_path: str = None,
):
    """
    训练并评估嵌入分类器

    使用前 train_count 条数据训练，后 test_count 条数据测试

    Args:
        data_path: 标注数据路径
        train_count: 训练样本数
        test_count: 测试样本数
        model_path: 模型路径
        save_path: 训练后保存类别嵌入的路径，如 "category_matrix.pt"
    """
    from collections import Counter

    print("=" * 70)
    print(" 嵌入分类器 - 训练与评估")
    print("=" * 70)
    print(f"数据文件: {data_path}")
    print(f"训练样本: {train_count} 条")
    print(f"测试样本: {test_count} 条")
    print("=" * 70)

    # 加载数据
    print("\n[1/4] 加载数据...")
    all_data = load_labeled_data(data_path)
    print(f"  总计: {len(all_data)} 条")

    # 划分训练集和测试集
    train_data = all_data[:train_count]
    test_data = all_data[train_count:train_count + test_count]
    print(f"  训练集: {len(train_data)} 条 (id 0-{train_count-1})")
    print(f"  测试集: {len(test_data)} 条 (id {train_count}-{train_count + test_count - 1})")

    # 创建分类器（不使用模板）
    print("\n[2/4] 加载模型...")
    classifier = create_classifier(model_path=model_path, use_template=False)

    # 训练
    print("\n[3/4] 训练...")
    sample_counts = classifier.train_from_data(train_data)

    # 保存类别嵌入
    if save_path:
        classifier.save_category_matrix(save_path)

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
                'confidence': result.confidence,
            })

    accuracy = correct / len(test_data) if test_data else 0

    # 打印结果
    print("\n" + "=" * 70)
    print(" 评估结果")
    print("=" * 70)
    print(f"\n准确率: {accuracy:.2%} ({correct}/{len(test_data)})")

    # 按类别统计
    per_class_metrics = {}
    print("\n各类别准确率:")
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

    # 显示部分错误样本
    if errors:
        print(f"\n错误样本 (共 {len(errors)} 条):")
        for err in errors[:10]:
            print(f"  [{err['id']:4d}] {err['title'][:40]:40s}")
            print(f"         真实: {err['true']}, 预测: {err['pred']} ({err['confidence']:.2f})")

    # 保存结果到 JSON
    from datetime import datetime
    result_dir = Path("data/results")
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    result_filename = f"eval_train{train_count}_test{test_count}_{timestamp}.json"
    result_path = result_dir / result_filename

    result_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'train_count': train_count,
            'test_count': test_count,
            'data_path': data_path,
            'model_path': model_path,
            'save_path': save_path,
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

    parser = argparse.ArgumentParser(description="网页分类器 - 训练与评估")
    parser.add_argument("--data", default="/home/zzh/webpage-classification/data/labeled/labeled.jsonl",
                       help="标注数据路径")
    parser.add_argument("--train", type=int, default=500, help="训练样本数")
    parser.add_argument("--test", type=int, default=100, help="测试样本数")
    parser.add_argument("--top-k", type=int, default=1,
                       help="Top-K 召回数量: 1=单阶段Embedding, 2/3=两阶段+Reranker (默认: 1)")
    parser.add_argument("--model", help="Embedding 模型路径")
    parser.add_argument("--save", help="保存类别嵌入的路径，不指定则自动命名")
    parser.add_argument("--load", help="加载已训练的类别嵌入")
    parser.add_argument("--no-save", action="store_true", help="不保存训练结果")
    parser.add_argument("--template", action="store_true", help="使用模板模式（不训练）")
    parser.add_argument("--doc-version", choices=["v1", "v2"], default="v1",
                       help="Reranker 类别描述版本: v1=详细列表, v2=简洁字符串 (默认: v1)")

    args = parser.parse_args()

    # 自动生成保存路径 (Embedding 矩阵与 top_k/doc_version 无关)
    if not args.load and not args.no_save and not args.save and not args.template:
        args.save = f"models/matrix_train{args.train}.pt"

    if args.template:
        # 模板模式：简单测试
        classifier = create_classifier(model_path=args.model, use_template=True)
        result = classifier.classify(
            title="中国人民银行宣布下调贷款利率",
            content="央行今日宣布，将一年期贷款市场报价利率下调10个基点..."
        )
        print(f"分类结果: {result.label}")
        print(f"置信度: {result.confidence:.2%}")
        print(f"各类别得分: {result.scores}")
        print(f"耗时: {result.latency_ms:.2f}ms")

    elif args.top_k == 1:
        # 单阶段 Embedding 模式
        if args.load:
            # 加载已训练的嵌入，直接测试
            print("=" * 70)
            print(" 单阶段 Embedding - 加载模式")
            print("=" * 70)

            all_data = load_labeled_data(args.data)
            test_data = all_data[args.train:args.train + args.test]
            print(f"测试集: {len(test_data)} 条")

            classifier = create_classifier(model_path=args.model, use_template=False)
            classifier.load_category_matrix(args.load)

            print("\n测试中...")
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
                save_path=args.save,
            )

    else:
        # 两阶段 Embedding + Reranker 模式
        # 设置日志
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"data/results/embedding_train{args.train}_topk{args.top_k}_{args.doc_version}_test{args.test}_{timestamp}.log"
        tee_logger = TeeLogger(log_filename)
        sys.stdout = tee_logger

        try:
            from two_stage_classifier import train_and_evaluate as two_stage_train_and_evaluate

            two_stage_train_and_evaluate(
                data_path=args.data,
                train_count=args.train,
                test_count=args.test,
                top_k=args.top_k,
                save_path=args.save,
                load_path=args.load,
                doc_version=args.doc_version,
                result_prefix="embedding",
            )
        finally:
            sys.stdout = tee_logger.terminal
            tee_logger.close()
            print(f"\n日志已保存到: {log_filename}")
