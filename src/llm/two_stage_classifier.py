"""
两阶段网页分类器
Stage 1: Embedding 召回 Top-K 候选
Stage 2: Reranker 精排选出 Top-1
"""

import torch
import time
import sys
import os
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime


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

# 添加路径
sys.path.append(str(Path(__file__).parent.parent.parent / "models" / "Qwen3-VL-Reranker-8B"))
sys.path.append(str(Path(__file__).parent.parent.parent / "models" / "Qwen3-VL-Reranker-8B" / "scripts"))

from embedding_classifier import (
    QwenEmbeddingClassifier,
    load_labeled_data,
    LABELS,
    CATEGORY_TEMPLATES,
)


@dataclass
class TwoStageResult:
    """两阶段分类结果"""
    label: str                      # 最终分类标签
    confidence: float               # 最终置信度
    topk_candidates: List[str]      # Stage1 的 topk 候选
    topk_scores: Dict[str, float]   # Stage1 的 topk 分数
    reranker_scores: Dict[str, float]  # Stage2 的重排分数
    latency_ms: float               # 总耗时


# Reranker 类别描述 v1：详细列表格式
RERANKER_DOCS_V1 = {
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

# Reranker 类别描述 v2：简洁字符串格式
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

# 版本映射
RERANKER_DOCS_VERSIONS = {
    "v1": RERANKER_DOCS_V1,  # 详细列表格式
    "v2": RERANKER_DOCS_V2,  # 简洁字符串格式
}


class TwoStageClassifier:
    """两阶段分类器"""

    def __init__(
        self,
        embedding_model_path: str = None,
        reranker_model_path: str = None,
        device: str = None,
        top_k: int = 3,
        doc_version: str = "v1",
    ):
        """
        初始化两阶段分类器

        Args:
            embedding_model_path: Embedding 模型路径
            reranker_model_path: Reranker 模型路径
            device: 设备
            top_k: Stage1 召回的候选数量
            doc_version: Reranker 类别描述版本 ("v1"=详细列表, "v2"=简洁字符串)
        """
        base_dir = Path(__file__).parent.parent.parent

        if embedding_model_path is None:
            embedding_model_path = str(base_dir / "models" / "Qwen3-VL-Embedding-8B")

        if reranker_model_path is None:
            reranker_model_path = str(base_dir / "models" / "Qwen3-VL-Reranker-8B")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.top_k = top_k
        self.doc_version = doc_version
        self.reranker_docs = RERANKER_DOCS_VERSIONS.get(doc_version, RERANKER_DOCS_V1)
        self.embedding_model_path = embedding_model_path
        self.reranker_model_path = reranker_model_path

        # Stage 1: 加载 Embedding 模型
        print("=" * 60)
        print(" 两阶段分类器初始化")
        print("=" * 60)
        print("\n[Stage 1] 加载 Embedding 模型...")
        self.embedder = QwenEmbeddingClassifier(
            model_path=embedding_model_path,
            device=device,
            use_template=True,  # 使用模板初始化
        )

        # Stage 2: 加载 Reranker 模型
        print("\n[Stage 2] 加载 Reranker 模型...")
        self._load_reranker(reranker_model_path)

        print("\n" + "=" * 60)
        print(" 初始化完成")
        print("=" * 60)

    def _load_reranker(self, model_path: str):
        """加载 Reranker 模型"""
        try:
            from qwen3_vl_reranker import Qwen3VLReranker

            try:
                self.reranker = Qwen3VLReranker(
                    model_name_or_path=model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2"
                )
                print("  Flash Attention 2 已启用")
            except Exception:
                print("  使用默认注意力机制")
                self.reranker = Qwen3VLReranker(
                    model_name_or_path=model_path,
                    torch_dtype=torch.bfloat16,
                )
            print(f"  Reranker 加载完成: {model_path}")
        except Exception as e:
            print(f"  Reranker 加载失败: {e}")
            self.reranker = None

    def train_from_data(self, train_data: List[Dict], batch_size: int = 32):
        """用标注数据训练 Embedding 模型"""
        return self.embedder.train_from_data(train_data, batch_size)

    def classify(
        self,
        title: str,
        content: str = "",
        url: str = "",
    ) -> TwoStageResult:
        """
        两阶段分类

        Args:
            title: 标题
            content: 内容
            url: URL

        Returns:
            TwoStageResult: 分类结果
        """
        start_time = time.time()

        # Stage 1: Embedding 召回 Top-K
        emb_result = self.embedder.classify(title=title, content=content, url=url)

        # 获取 Top-K 候选
        sorted_scores = sorted(emb_result.scores.items(), key=lambda x: -x[1])
        top_k_candidates = [label for label, _ in sorted_scores[:self.top_k]]
        top_k_scores = {label: score for label, score in sorted_scores[:self.top_k]}

        # Stage 2: Reranker 精排
        if self.reranker is None or len(top_k_candidates) <= 1:
            # 没有 Reranker 或只有一个候选，直接返回
            final_label = top_k_candidates[0]
            final_confidence = emb_result.confidence
            reranker_scores = {}
        else:
            # 构建 query
            query_text = f"Title: {title}\nContent: {content[:1024]}"

            # 构建候选 documents（只对 top-k 候选进行重排）
            # 根据版本处理：v1 是列表需要 join，v2 是字符串直接用
            def get_doc_text(label):
                doc = self.reranker_docs[label]
                if isinstance(doc, list):
                    return "；".join(doc)  # v1: 列表 -> 字符串
                return doc  # v2: 直接返回字符串

            candidate_docs = [{"text": get_doc_text(label)} for label in top_k_candidates]

            # Reranker 输入
            reranker_input = {
                "instruction": "你是一个支持多种语言的新闻分类专家，判断这篇新闻属于哪个类别。如果遇到非简体中文的文本，必须先翻译为简体中文再进行分类。",
                "query": {"text": query_text},
                "documents": candidate_docs,
            }

            # 获取分数
            scores = self.reranker.process(reranker_input)
            reranker_scores = {label: score for label, score in zip(top_k_candidates, scores)}

            # 选择最高分
            best_idx = scores.index(max(scores))
            final_label = top_k_candidates[best_idx]
            final_confidence = scores[best_idx]

        latency_ms = (time.time() - start_time) * 1000

        return TwoStageResult(
            label=final_label,
            confidence=final_confidence,
            topk_candidates=top_k_candidates,
            topk_scores=top_k_scores,
            reranker_scores=reranker_scores,
            latency_ms=latency_ms,
        )

    def batch_classify(
        self,
        items: List[Dict],
        batch_size: int = 1,
    ) -> List[TwoStageResult]:
        """
        批量分类

        Args:
            items: 输入列表
            batch_size: 批大小（Reranker 目前逐条处理）

        Returns:
            List[TwoStageResult]: 分类结果列表
        """
        results = []
        print(f"\n  {'='*60}")
        print(f"  两阶段分类 (共 {len(items)} 条)")
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
            print(f"         Top{self.top_k}: [{topk_str}] → {result.label} ({result.confidence:.2f})")

            if (idx + 1) % 10 == 0:
                print(f"  --- 进度: {idx + 1}/{len(items)} ---")

        print(f"  {'='*60}\n")
        return results


def train_and_evaluate(
    data_path: str,
    train_count: int = 500,
    test_count: int = 100,
    top_k: int = 2,
    save_path: str = None,
    load_path: str = None,
    doc_version: str = "v1",
    result_prefix: str = None,
):
    """
    训练并评估两阶段分类器

    Args:
        data_path: 标注数据路径
        train_count: 训练样本数
        test_count: 测试样本数
        top_k: Stage1 召回数量
        save_path: 训练后保存类别嵌入的路径
        load_path: 加载已训练的类别嵌入，跳过训练
        doc_version: Reranker 类别描述版本 ("v1"=详细列表, "v2"=简洁字符串)
    """
    print("=" * 70)
    print(" 两阶段分类器 - 训练与评估")
    print(f" Stage 1: Embedding (Top-{top_k}) + Stage 2: Reranker (Top-1)")
    print("=" * 70)
    print(f"数据文件: {data_path}")
    print(f"训练样本: {train_count} 条")
    print(f"测试样本: {test_count} 条")
    print(f"Top-K: {top_k}")
    print(f"Doc版本: {doc_version} ({'详细列表' if doc_version == 'v1' else '简洁字符串'})")
    if save_path:
        print(f"保存路径: {save_path}")
    if load_path:
        print(f"加载路径: {load_path}")
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
    classifier = TwoStageClassifier(top_k=top_k, doc_version=doc_version)

    # 训练或加载 Embedding
    if load_path:
        print("\n[3/5] 加载已训练的类别嵌入...")
        classifier.embedder.load_category_matrix(load_path)
    else:
        print("\n[3/5] 训练 Embedding...")
        classifier.train_from_data(train_data)
        if save_path:
            classifier.embedder.save_category_matrix(save_path)

    # 测试
    print("\n[4/5] 测试...")
    results = classifier.batch_classify(test_data)

    # 评估
    print("\n[5/5] 评估...")

    # 统计指标
    emb_top1_correct = 0      # Embedding Top-1 正确数 (不经过 Reranker)
    reranker_correct = 0      # Reranker 最终选择正确数 (Top-K)
    topk_recall = 0           # Top-K 召回数 (真实标签在候选中)
    reranker_fixed = 0        # Reranker 改对: Emb 错 → Reranker 对
    reranker_broke = 0        # Reranker 改错: Emb 对 → Reranker 错
    errors = []

    # Top-1/2/3/.../K 各级召回统计
    topk_recall_counts = {k: 0 for k in range(1, top_k + 1)}
    # Top-2/3/.../K 各级 Reranker 准确率统计
    topk_reranker_correct = {k: 0 for k in range(2, top_k + 1)}

    for item, result in zip(test_data, results):
        true_label = item.get("label")
        pred_label = result.label
        emb_top1 = result.topk_candidates[0]  # Embedding 的第一名

        emb_correct = (true_label == emb_top1)
        rrk_correct = (true_label == pred_label)
        in_topk = (true_label in result.topk_candidates)

        # 计算 Top-1/2/3/.../K 各级召回
        for k in range(1, top_k + 1):
            if true_label in result.topk_candidates[:k]:
                topk_recall_counts[k] += 1

        # 计算 Top-2/3/.../K 各级 Reranker 准确率
        # 利用已有的 reranker_scores，取 Top-k 候选中得分最高的
        for k in range(2, top_k + 1):
            top_k_candidates = result.topk_candidates[:k]
            if result.reranker_scores:
                # 在 Top-k 候选中找 Reranker 得分最高的
                top_k_scores = {label: result.reranker_scores.get(label, 0)
                               for label in top_k_candidates}
                reranker_pick = max(top_k_scores, key=top_k_scores.get)
            else:
                # 没有 Reranker 分数，取 Embedding 第一名
                reranker_pick = top_k_candidates[0]

            if true_label == reranker_pick:
                topk_reranker_correct[k] += 1

        # Top-1 准确率 (仅 Embedding，不用 Reranker)
        if emb_correct:
            emb_top1_correct += 1

        # Top-K 准确率 (Reranker 最终选择)
        if rrk_correct:
            reranker_correct += 1

        # Top-K 召回率 (真实标签是否在 Top-K 候选中)
        if in_topk:
            topk_recall += 1
            # Reranker 改对/改错分析 (仅在 Top-K 召回时有意义)
            if not emb_correct and rrk_correct:
                reranker_fixed += 1  # Emb 第一名错，但 Reranker 选对了
            elif emb_correct and not rrk_correct:
                reranker_broke += 1  # Emb 第一名对，但 Reranker 选错了
        else:
            errors.append({
                "id": item.get("id"),
                "title": item.get("title", "")[:50],
                "true": true_label,
                "pred": pred_label,
                "topk": result.topk_candidates,
                "confidence": result.confidence,
            })

    total = len(test_data)
    emb_top1_acc = emb_top1_correct / total if total else 0
    reranker_acc = reranker_correct / total if total else 0
    topk_recall_rate = topk_recall / total if total else 0

    # 计算各级召回率
    topk_recall_rates = {k: topk_recall_counts[k] / total if total else 0 for k in range(1, top_k + 1)}
    # 计算各级 Reranker 准确率
    topk_reranker_acc = {k: topk_reranker_correct[k] / total if total else 0 for k in range(2, top_k + 1)}

    # 打印结果
    print("\n" + "=" * 70)
    print(" 评估结果")
    print("=" * 70)

    # 动态生成表格（包含各级 Reranker 准确率）
    print(f"\n  ┌─────────────┬────────────┬────────────┐")
    print(f"  │   指标      │   准确率   │   召回率   │")
    print(f"  ├─────────────┼────────────┼────────────┤")
    print(f"  │ Top-1 (Emb) │ {emb_top1_acc:>8.2%}   │ {topk_recall_rates[1]:>8.2%}   │")
    for k in range(2, top_k + 1):
        print(f"  │ Top-{k} +Rrk  │ {topk_reranker_acc[k]:>8.2%}   │ {topk_recall_rates[k]:>8.2%}   │")
    print(f"  └─────────────┴────────────┴────────────┘")

    print(f"\n  Top-K 召回详情:")
    for k in range(1, top_k + 1):
        print(f"    - Top-{k} 召回: {topk_recall_counts[k]}/{total} ({topk_recall_rates[k]:.2%})")

    print(f"\n  Top-K + Reranker 准确率详情:")
    print(f"    - Top-1 (仅Emb): {emb_top1_correct}/{total} ({emb_top1_acc:.2%})")
    for k in range(2, top_k + 1):
        net_gain = topk_reranker_correct[k] - emb_top1_correct
        print(f"    - Top-{k} +Rrk: {topk_reranker_correct[k]}/{total} ({topk_reranker_acc[k]:.2%}) [净收益: {net_gain:+d}]")

    # 各类别准确率
    per_class_metrics = {}
    print("\n  各类别准确率:")
    for label in LABELS:
        label_items = [(item, result) for item, result in zip(test_data, results)
                       if item.get("label") == label]
        if label_items:
            label_correct = sum(1 for item, result in label_items
                               if item.get("label") == result.label)
            label_acc = label_correct / len(label_items)
            per_class_metrics[label] = {
                'accuracy': label_acc,
                'correct': label_correct,
                'total': len(label_items),
            }
            print(f"    {label}: {label_acc:.2%} ({label_correct}/{len(label_items)})")

    # 错误分析
    if errors:
        print(f"\n  Top-{top_k} 未召回的样本 (共 {len(errors)} 条):")
        for err in errors[:5]:
            print(f"    [{err['id']:4d}] {err['title'][:40]}")
            print(f"           真实: {err['true']}, Top{top_k}: {err['topk']}")

    # 保存结果到 JSON
    import json
    from datetime import datetime
    from pathlib import Path

    result_dir = Path("/home/zzh/webpage-classification/data/results")
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if result_prefix:
        result_filename = f"{result_prefix}_train{train_count}_topk{top_k}_{doc_version}_test{test_count}_{timestamp}.json"
    else:
        result_filename = f"eval_train{train_count}_topk{top_k}_{doc_version}_test{test_count}_{timestamp}.json"
    result_path = result_dir / result_filename

    # 构建各级 Top-K 召回数据
    topk_recall_data = {}
    for k in range(1, top_k + 1):
        topk_recall_data[f'top{k}_recall'] = topk_recall_rates[k]
        topk_recall_data[f'top{k}_recall_count'] = topk_recall_counts[k]

    # 构建各级 Top-K + Reranker 准确率数据
    topk_reranker_data = {}
    for k in range(2, top_k + 1):
        topk_reranker_data[f'top{k}_reranker_accuracy'] = topk_reranker_acc[k]
        topk_reranker_data[f'top{k}_reranker_correct'] = topk_reranker_correct[k]
        topk_reranker_data[f'top{k}_reranker_net_gain'] = topk_reranker_correct[k] - emb_top1_correct

    result_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'train_count': train_count,
            'test_count': test_count,
            'top_k': top_k,
            'doc_version': doc_version,
            'data_path': data_path,
            'save_path': save_path,
            'load_path': load_path,
        },
        'overall': {
            'total': total,
            # Top-1 准确率 (仅 Embedding)
            'top1_accuracy': emb_top1_acc,
            'top1_correct': emb_top1_correct,
            # Top-K 各级召回率
            **topk_recall_data,
            # Top-K 各级 Reranker 准确率
            **topk_reranker_data,
        },
        'per_class': per_class_metrics,
        'errors': errors,
    }

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"\n  结果已保存至: {result_path}")
    print("=" * 70)

    return {
        'top1_accuracy': emb_top1_acc,
        'topk_recall_rates': topk_recall_rates,  # {1: 0.66, 2: 0.85, 3: 0.92, 4: 0.96}
        'topk_reranker_acc': topk_reranker_acc,  # {2: 0.70, 3: 0.68, 4: 0.63}
        'errors': errors,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="两阶段分类器训练与评估")
    parser.add_argument("--data", default="/home/zzh/webpage-classification/data/labeled/labeled.jsonl",
                       help="标注数据路径")
    parser.add_argument("--train", type=int, default=500, help="训练样本数")
    parser.add_argument("--test", type=int, default=100, help="测试样本数")
    parser.add_argument("--top-k", type=int, default=2, help="Stage1 召回数量 (默认: 2)")
    parser.add_argument("--doc-version", choices=["v1", "v2"], default="v1",
                       help="Reranker 类别描述版本: v1=详细列表, v2=简洁字符串 (默认: v1)")
    parser.add_argument("--save", help="训练后保存类别嵌入的路径，不指定则自动命名")
    parser.add_argument("--load", help="加载已训练的类别嵌入，跳过训练")
    parser.add_argument("--no-save", action="store_true", help="不保存训练结果")

    args = parser.parse_args()

    # 自动生成保存路径 (Embedding 矩阵与 top_k/doc_version 无关)
    if not args.load and not args.no_save and not args.save:
        args.save = f"models/matrix_train{args.train}.pt"

    # 自动保存日志（包含 doc_version）
    result_dir = Path("/home/zzh/webpage-classification/data/results")
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"eval_train{args.train}_topk{args.top_k}_{args.doc_version}_test{args.test}_{timestamp}.log"
    log_path = result_dir / log_filename
    logger = TeeLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    print(f"Log 保存至: {log_path}")

    try:
        train_and_evaluate(
            data_path=args.data,
            train_count=args.train,
            test_count=args.test,
            top_k=args.top_k,
            save_path=args.save,
            load_path=args.load,
            doc_version=args.doc_version,
        )
    finally:
        # 恢复标准输出并关闭日志
        sys.stdout = logger.terminal
        sys.stderr = logger.terminal
        logger.close()
        print(f"\nLog 已保存至: {log_path}")
