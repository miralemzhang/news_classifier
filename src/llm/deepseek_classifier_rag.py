"""
RAG 检索增强网页分类器
Stage 1: Embedding 召回 Top-2 候选
Stage 2: 从训练集检索相似样本，辅助 DeepSeek 决策
"""

import torch
import torch.nn.functional as F
import time
import re
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
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
class RAGClassifyResult:
    """RAG 分类结果"""
    label: str                      # 最终分类标签
    confidence: float               # 置信度
    top2_candidates: List[str]      # Stage1 的 top2 候选
    top2_scores: Dict[str, float]   # Stage1 的 top2 分数
    retrieved_examples: List[Dict]  # 检索到的相似样本
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
    ],
    "经济": [
        "股票市场行情，基金理财，银行利率调整，金融市场波动",
        "企业财报，上市公司，投资并购，公司业绩，商业新闻",
        "房价走势，物价指数，GDP增长数据，经济增长率统计",
        "贸易进出口数据，国际贸易额，进出口统计",
    ],
    "军事": [
        "军队演习，武器装备，国防建设",
        "战争冲突，军事行动，武装冲突，军事打击，战场战况",
        "军舰航母，战斗机，导弹，坦克，军事科技，武器系统",
        "军事部署，边境武装对峙，前线战况，部队调动",
    ],
    "社会": [
        "民生新闻，社会热点，公共事件，市民生活",
        "教育医疗，交通出行，社会保障，公共服务",
        "列车事故，车祸，空难，沉船，工业事故，安全事故",
        "犯罪案件，社会治安，法律诉讼，民事纠纷，刑事案件",
    ],
    "科技": [
        "人工智能，互联网科技，手机数码产品，科技公司",
        "科学研究，技术创新，航天航空，太空探索",
        "新能源汽车，芯片半导体，5G通信，电动车",
        "软件应用，云计算，大数据，网络安全",
    ],
    "体育": [
        "足球篮球比赛，运动员，体育赛事，比赛结果",
        "奥运会，世界杯，NBA，CBA联赛，欧冠",
        "健身运动，马拉松，电子竞技，极限运动",
        "体育明星，球队转会，体育评论，赛事分析",
    ],
    "娱乐": [
        "明星八卦，影视综艺，音乐演唱会，娱乐新闻",
        "电影电视剧，选秀节目，网红直播",
        "游戏动漫，时尚潮流，颁奖典礼",
        "娱乐圈，粉丝文化，偶像团体",
    ],
    "其他": [
        "生活百科，美食旅游，健康养生",
        "天气预报，星座运势，历史文化",
        "日常杂谈，情感故事，职场经验，个人博客",
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


class RAGClassifier:
    """RAG 检索增强分类器"""

    def __init__(
        self,
        embedding_model_path: str = None,
        deepseek_model_path: str = None,
        device: str = None,
        top_k: int = 2,
        retrieve_k: int = 3,
        doc_version: str = "v1",
    ):
        """
        初始化 RAG 分类器

        Args:
            embedding_model_path: Embedding 模型路径
            deepseek_model_path: DeepSeek 模型路径
            device: 设备
            top_k: Stage1 召回的候选类别数量
            retrieve_k: 从训练集检索的相似样本数量
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
        self.retrieve_k = retrieve_k
        self.doc_version = doc_version
        self.category_docs = CATEGORY_DOCS_VERSIONS.get(doc_version, CATEGORY_DESCRIPTIONS_V1)

        # 训练样本嵌入存储: {id: (embedding, label, title, content)}
        self.train_embeddings: Dict[int, Tuple[torch.Tensor, str, str, str]] = {}

        # Stage 1: 加载 Embedding 模型
        print("=" * 60)
        print(" RAG 检索增强分类器")
        print("=" * 60)
        print(f"  类别描述版本: {doc_version}")
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
        """
        用标注数据训练 Embedding 模型，并存储训练样本嵌入

        Args:
            train_data: 训练数据
            batch_size: 批大小
        """
        # 1. 训练 Embedding 类别中心
        self.embedder.train_from_data(train_data, batch_size)

        # 2. 计算并存储所有训练样本的嵌入
        print("\n[RAG] 计算训练样本嵌入...")
        self.train_embeddings.clear()

        for idx, item in enumerate(train_data):
            item_id = item.get("id", idx)
            title = item.get("title", "")
            content = item.get("text", item.get("content", ""))
            label = item.get("label", "")

            # 计算嵌入
            text = f"{title} {content[:512]}" if content else title
            embedding = self.embedder.get_text_embedding(text)

            # 存储
            self.train_embeddings[item_id] = (embedding, label, title, content[:200])

            if (idx + 1) % 100 == 0:
                print(f"  已处理 {idx + 1}/{len(train_data)} 条训练样本")

        print(f"  训练样本嵌入存储完成，共 {len(self.train_embeddings)} 条")

    def retrieve_similar(
        self,
        query_embedding: torch.Tensor,
        candidate_labels: List[str],
        k: int = 3,
    ) -> List[Dict]:
        """
        从训练集检索与查询最相似的 K 个样本

        Args:
            query_embedding: 查询文本的嵌入
            candidate_labels: 限定在这些候选类别内检索
            k: 返回的样本数量

        Returns:
            检索到的相似样本列表
        """
        if not self.train_embeddings:
            return []

        # 筛选候选类别内的训练样本
        candidates = []
        for item_id, (emb, label, title, content) in self.train_embeddings.items():
            if label in candidate_labels:
                candidates.append({
                    "id": item_id,
                    "embedding": emb,
                    "label": label,
                    "title": title,
                    "content": content,
                })

        if not candidates:
            return []

        # 计算相似度
        candidate_embeddings = torch.stack([c["embedding"] for c in candidates])
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            candidate_embeddings,
            dim=1
        )

        # 取 Top-K
        top_k_indices = similarities.argsort(descending=True)[:k]

        retrieved = []
        for idx in top_k_indices:
            c = candidates[idx]
            retrieved.append({
                "id": c["id"],
                "label": c["label"],
                "title": c["title"],
                "content": c["content"],
                "similarity": similarities[idx].item(),
            })

        return retrieved

    def _build_rag_prompt(
        self,
        title: str,
        content: str,
        candidates: List[str],
        retrieved_examples: List[Dict],
    ) -> str:
        """构建 RAG 增强的 prompt"""
        # 获取类别描述文本（支持 v1 列表格式和 v2 字符串格式）
        def get_doc_text(label):
            doc = self.category_docs[label]
            if isinstance(doc, list):
                return "；".join(doc)  # v1: 列表 -> 分号连接
            return doc  # v2: 直接返回字符串

        # 候选类别描述
        candidate_desc = "\n".join([
            f"  - {cat}: {get_doc_text(cat)}"
            for cat in candidates
        ])

        # 检索到的相似样本
        if retrieved_examples:
            examples_text = "\n".join([
                f"  - 「{ex['title'][:50]}」→ {ex['label']}"
                for ex in retrieved_examples
            ])
        else:
            examples_text = "  无"

        prompt = f"""你是一个新闻分类专家。请根据以下参考示例和候选类别，对新闻进行分类。

【参考相似新闻的分类】
{examples_text}

【候选类别】
{candidate_desc}

【待分类新闻】
标题: {title}
内容: {content[:300] if content else "无"}

请根据参考示例的分类规律，选择最合适的类别。只需回答类别名称，如"时政"或"经济"。

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
    ) -> RAGClassifyResult:
        """
        RAG 增强分类

        Args:
            title: 标题
            content: 内容
            url: URL

        Returns:
            RAGClassifyResult: 分类结果
        """
        start_time = time.time()

        # Stage 1: Embedding 召回 Top-K 候选类别
        emb_result = self.embedder.classify(title=title, content=content, url=url)

        # 获取 Top-K 候选
        sorted_scores = sorted(emb_result.scores.items(), key=lambda x: -x[1])
        top_k_candidates = [label for label, _ in sorted_scores[:self.top_k]]
        top_k_scores = {label: score for label, score in sorted_scores[:self.top_k]}

        # 获取查询文本的嵌入
        query_text = f"{title} {content[:512]}" if content else title
        query_embedding = self.embedder.get_text_embedding(query_text)

        # Stage 2: 从训练集检索相似样本
        retrieved_examples = self.retrieve_similar(
            query_embedding,
            top_k_candidates,
            k=self.retrieve_k,
        )

        # Stage 3: DeepSeek 基于检索结果做决策
        if len(top_k_candidates) <= 1:
            # 只有一个候选，直接返回
            final_label = top_k_candidates[0]
            thinking = ""
            raw_output = ""
        else:
            # 构建 RAG prompt
            prompt = self._build_rag_prompt(title, content, top_k_candidates, retrieved_examples)

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

        return RAGClassifyResult(
            label=final_label,
            confidence=top_k_scores.get(final_label, 0.5),
            top2_candidates=top_k_candidates,
            top2_scores=top_k_scores,
            retrieved_examples=retrieved_examples,
            thinking=thinking,
            raw_output=raw_output if len(top_k_candidates) > 1 else "",
            latency_ms=latency_ms,
        )

    def batch_classify(
        self,
        items: List[Dict],
    ) -> List[RAGClassifyResult]:
        """批量分类"""
        results = []
        print(f"\n  {'='*60}")
        print(f"  RAG 检索增强分类 (共 {len(items)} 条)")
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
            retrieved_labels = [ex["label"] for ex in result.retrieved_examples]
            print(f"  [{item_id:4d}] {title[:35]:35s}")
            print(f"         Top2: [{top2_str}] | 检索: {retrieved_labels} → {result.label}")

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

    def save_train_embeddings(self, path: str):
        """保存训练样本嵌入"""
        data = {
            item_id: {
                "embedding": emb.cpu().numpy().tolist(),
                "label": label,
                "title": title,
                "content": content,
            }
            for item_id, (emb, label, title, content) in self.train_embeddings.items()
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"  训练样本嵌入已保存至: {path}")

    def load_train_embeddings(self, path: str):
        """加载训练样本嵌入"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.train_embeddings.clear()
        for item_id, item in data.items():
            emb = torch.tensor(item["embedding"], device=self.device)
            self.train_embeddings[int(item_id)] = (
                emb,
                item["label"],
                item["title"],
                item["content"],
            )
        print(f"  训练样本嵌入已加载，共 {len(self.train_embeddings)} 条")


def train_and_evaluate(
    data_path: str,
    train_count: int = 500,
    test_count: int = 100,
    top_k: int = 2,
    retrieve_k: int = 3,
    doc_version: str = "v1",
    save_path: str = None,
    load_path: str = None,
    save_embeddings_path: str = None,
    load_embeddings_path: str = None,
):
    """
    训练并评估 RAG 分类器

    Args:
        data_path: 标注数据路径
        train_count: 训练样本数
        test_count: 测试样本数
        top_k: Stage1 召回的候选类别数量
        retrieve_k: 检索的相似样本数量
        doc_version: 类别描述版本 (v1=详细列表, v2=简洁字符串)
        save_path: 保存类别嵌入的路径
        load_path: 加载类别嵌入的路径
        save_embeddings_path: 保存训练样本嵌入的路径
        load_embeddings_path: 加载训练样本嵌入的路径
    """
    print("=" * 70)
    print(" RAG 检索增强分类器 - 训练与评估")
    print(f" Stage 1: Embedding (Top-{top_k})")
    print(f" Stage 2: RAG 检索 (K={retrieve_k}) + DeepSeek")
    print("=" * 70)
    print(f"数据文件: {data_path}")
    print(f"训练样本: {train_count} 条")
    print(f"测试样本: {test_count} 条")
    print(f"检索数量: {retrieve_k} 条")
    print(f"类别描述: {doc_version}")
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
    classifier = RAGClassifier(top_k=top_k, retrieve_k=retrieve_k, doc_version=doc_version)

    # 训练或加载 Embedding
    if load_path and load_embeddings_path:
        print("\n[3/5] 加载已训练的模型...")
        classifier.load_category_matrix(load_path)
        classifier.load_train_embeddings(load_embeddings_path)
    else:
        print("\n[3/5] 训练 Embedding 并构建检索库...")
        classifier.train_from_data(train_data)
        if save_path:
            classifier.save_category_matrix(save_path)
        if save_embeddings_path:
            classifier.save_train_embeddings(save_embeddings_path)

    # 测试
    print("\n[4/5] 测试...")
    results = classifier.batch_classify(test_data)

    # 评估
    print("\n[5/5] 评估...")
    correct = 0
    topk_recall = 0
    rag_helped = 0  # RAG 帮助改对
    rag_hurt = 0    # RAG 导致改错
    errors = []

    for item, result in zip(test_data, results):
        true_label = item.get("label")
        pred_label = result.label
        emb_top1 = result.top2_candidates[0]  # Embedding 的第一名

        emb_correct = (true_label == emb_top1)
        rag_correct = (true_label == pred_label)

        if rag_correct:
            correct += 1

        if true_label in result.top2_candidates:
            topk_recall += 1
            # 分析 RAG 的效果
            if not emb_correct and rag_correct:
                rag_helped += 1
            elif emb_correct and not rag_correct:
                rag_hurt += 1
        else:
            errors.append({
                "id": item.get("id"),
                "title": item.get("title", "")[:50],
                "true": true_label,
                "pred": pred_label,
                "top2": result.top2_candidates,
                "retrieved": [ex["label"] for ex in result.retrieved_examples],
            })

    accuracy = correct / len(test_data) if test_data else 0
    recall_at_k = topk_recall / len(test_data) if test_data else 0

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
    print(f"\n  Top-1 准确率: {accuracy:.2%} ({correct}/{len(test_data)})")
    print(f"  Top-{top_k} 召回率: {recall_at_k:.2%} ({topk_recall}/{len(test_data)})")
    print(f"\n  RAG 效果分析:")
    print(f"    - RAG 改对: {rag_helped} 条 (Emb错→RAG对)")
    print(f"    - RAG 改错: {rag_hurt} 条 (Emb对→RAG错)")
    print(f"    - RAG 净收益: {rag_helped - rag_hurt} 条")

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
    result_filename = f"deepseek_rag_train{train_count}_topk{top_k}_doc_{doc_version}_test{test_count}_{timestamp}.json"
    result_path = result_dir / result_filename

    result_data = {
        'timestamp': timestamp,
        'config': {
            'method': 'RAG (Embedding + Retrieval + DeepSeek)',
            'train_count': train_count,
            'test_count': test_count,
            'top_k': top_k,
            'retrieve_k': retrieve_k,
            'doc_version': doc_version,
        },
        'overall': {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(test_data),
            f'top{top_k}_recall': recall_at_k,
            'rag_helped': rag_helped,
            'rag_hurt': rag_hurt,
            'rag_net_benefit': rag_helped - rag_hurt,
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

    parser = argparse.ArgumentParser(description="RAG 检索增强分类器")
    parser.add_argument("--data", default="/home/zzh/webpage-classification/data/labeled/labeled.jsonl",
                       help="标注数据路径")
    parser.add_argument("--train", type=int, default=500, help="训练样本数")
    parser.add_argument("--test", type=int, default=100, help="测试样本数")
    parser.add_argument("--top-k", type=int, default=2, help="Stage1 召回数量")
    parser.add_argument("--retrieve-k", type=int, default=3, help="检索相似样本数量")
    parser.add_argument("--doc-version", choices=["v1", "v2"], default="v1",
                       help="类别描述版本 (v1=详细列表, v2=简洁字符串)")
    parser.add_argument("--save", help="保存类别嵌入的路径")
    parser.add_argument("--load", help="加载已训练的类别嵌入")
    parser.add_argument("--save-embeddings", help="保存训练样本嵌入的路径")
    parser.add_argument("--load-embeddings", help="加载训练样本嵌入")

    args = parser.parse_args()

    # 自动生成保存路径 (Embedding 矩阵与 top_k/doc_version 无关)
    if not args.load and not args.save:
        args.save = f"../../models/matrix_train{args.train}.pt"
    # RAG 训练样本嵌入（RAG 专用）
    if not args.load_embeddings and not args.save_embeddings:
        args.save_embeddings = f"../../models/rag_train_embeddings_{args.train}.json"

    # 自动保存 log
    result_dir = Path("/home/zzh/webpage-classification/data/results")
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = result_dir / f"deepseek_rag_train{args.train}_topk{args.top_k}_doc_{args.doc_version}_test{args.test}_{timestamp}.log"
    logger = TeeLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    print(f"Log 保存至: {log_path}")

    train_and_evaluate(
        data_path=args.data,
        train_count=args.train,
        test_count=args.test,
        top_k=args.top_k,
        retrieve_k=args.retrieve_k,
        doc_version=args.doc_version,
        save_path=args.save,
        load_path=args.load,
        save_embeddings_path=args.save_embeddings,
        load_embeddings_path=args.load_embeddings,
    )
