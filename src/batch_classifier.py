"""
批量分类器
两阶段分类：规则分类 → LLM 兜底
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from tqdm import tqdm

# 导入规则分类器
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from rules.classifier import RuleClassifier


@dataclass
class ClassificationResult:
    """分类结果"""
    id: str
    title: str
    url: str
    category: str
    confidence: float
    method: str  # 'rule' 或 'llm'
    matched_keywords: list
    needs_review: bool = False


class BatchClassifier:
    """批量分类器"""

    def __init__(self, keywords_path: str = None, use_llm: bool = True, llm_model: str = "qwen2:7b",
                 prompt_path: str = None):
        self.rule_classifier = RuleClassifier(keywords_path)
        self.use_llm = use_llm
        self.llm_model = llm_model

        # LLM 分类器（延迟加载）
        self._llm_classifier = None

        # 加载 prompt 模板
        if prompt_path is None:
            base_dir = Path(__file__).parent.parent
            prompt_path = base_dir / "prompts" / "classification_prompt.txt"

        self.prompt_template = None
        if Path(prompt_path).exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
            print(f"✓ 已加载 prompt 模板: {prompt_path}")

        # 统计
        self.stats = {
            "total": 0,
            "rule_classified": 0,
            "llm_classified": 0,
            "other": 0,
            "by_category": {}
        }
    
    @property
    def llm_classifier(self):
        """延迟加载 LLM 分类器"""
        if self._llm_classifier is None and self.use_llm:
            try:
                import ollama
                self._llm_classifier = ollama.Client()
                print(f"✓ LLM 分类器已加载: {self.llm_model}")
            except ImportError:
                print("⚠ ollama 未安装，将跳过 LLM 分类")
                self.use_llm = False
            except Exception as e:
                print(f"⚠ LLM 加载失败: {e}")
                self.use_llm = False
        return self._llm_classifier
    
    def classify_single(self, title: str, url: str = "", content: str = "", doc_id: str = "") -> ClassificationResult:
        """
        分类单个文档
        1. 先用规则分类
        2. 规则无法分类则用 LLM
        3. LLM 也无法确定则归为"其他"
        """
        self.stats["total"] += 1
        
        # 阶段1：规则分类
        rule_result = self.rule_classifier.classify(
            title=title,
            content=content,
            url_path=url
        )
        
        if rule_result and rule_result.confidence >= 0.7:
            # 规则分类成功
            category = rule_result.label
            confidence = rule_result.confidence
            method = "rule"
            matched = rule_result.matched_keywords
            self.stats["rule_classified"] += 1
        else:
            # 阶段2：LLM 分类
            if self.use_llm and self.llm_classifier:
                llm_result = self._classify_with_llm(title, url, content)
                if llm_result:
                    category = llm_result["label"]
                    confidence = llm_result["confidence"]
                    method = "llm"
                    matched = []
                    self.stats["llm_classified"] += 1
                else:
                    category = "其他"
                    confidence = 0.3
                    method = "fallback"
                    matched = []
                    self.stats["other"] += 1
            else:
                # 无 LLM，直接归其他
                category = "其他"
                confidence = 0.3
                method = "no_match"
                matched = []
                self.stats["other"] += 1
        
        # 更新类别统计
        self.stats["by_category"][category] = self.stats["by_category"].get(category, 0) + 1
        
        return ClassificationResult(
            id=doc_id,
            title=title,
            url=url,
            category=category,
            confidence=confidence,
            method=method,
            matched_keywords=matched,
            needs_review=(confidence < 0.7)
        )
    
    def _classify_with_llm(self, title: str, url: str, content: str) -> Optional[dict]:
        """使用 LLM 分类"""
        # 使用模板或默认 prompt
        if self.prompt_template:
            prompt = self.prompt_template.format(
                title=title,
                url=url,
                content=content[:500] if content else ""
            )
        else:
            # 默认简单 prompt
            prompt = f"""你是一个新闻分类专家。请将以下新闻分类到8个类别之一。

类别：时政、经济、军事、社会、科技、体育、娱乐、其他

新闻标题：{title}

请直接回答类别名称（只需要回答一个词）："""

        try:
            response = self.llm_classifier.generate(
                model=self.llm_model,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 20}
            )

            result_text = response['response'].strip()

            # 解析结果
            categories = ["时政", "经济", "军事", "社会", "科技", "体育", "娱乐", "其他"]
            for cat in categories:
                if cat in result_text:
                    return {"label": cat, "confidence": 0.75}

            return {"label": "其他", "confidence": 0.5}

        except Exception as e:
            print(f"LLM 分类失败: {e}")
            return None
    
    def classify_batch(self, data: list, show_progress: bool = True) -> list:
        """
        批量分类
        
        Args:
            data: [{"id": "...", "title": "...", "url": "...", "content": "..."}, ...]
        
        Returns:
            分类结果列表
        """
        results = []
        
        iterator = tqdm(data, desc="分类中") if show_progress else data

        for item in iterator:
            # 兼容不同的字段名格式
            result = self.classify_single(
                title=item.get("title", ""),
                url=item.get("articleLink", item.get("url", "")),      # 兼容 articleLink
                content=item.get("text", item.get("content", "")),    # 兼容 text
                doc_id=item.get("id", "")
            )
            results.append(result)
        
        return results
    
    def classify_file(self, input_file: str, output_file: str) -> dict:
        """
        分类文件中的数据
        
        Args:
            input_file: 输入文件（JSONL 格式）
            output_file: 输出文件（JSONL 格式）
        
        Returns:
            统计信息
        """
        # 读取数据
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                if 'id' not in item:
                    item['id'] = str(i)
                data.append(item)
        
        print(f"读取 {len(data)} 条数据")
        
        # 分类
        results = self.classify_batch(data)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + '\n')
        
        print(f"结果已保存至: {output_file}")
        
        # 返回统计
        return self.get_stats()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return self.stats
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 50)
        print("分类统计")
        print("=" * 50)
        print(f"总数: {self.stats['total']}")
        print(f"规则分类: {self.stats['rule_classified']} ({self.stats['rule_classified']/max(1,self.stats['total'])*100:.1f}%)")
        print(f"LLM 分类: {self.stats['llm_classified']} ({self.stats['llm_classified']/max(1,self.stats['total'])*100:.1f}%)")
        print(f"其他: {self.stats['other']} ({self.stats['other']/max(1,self.stats['total'])*100:.1f}%)")
        print("\n各类别分布:")
        for cat, count in sorted(self.stats['by_category'].items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count} ({count/max(1,self.stats['total'])*100:.1f}%)")


def demo():
    """演示功能"""
    classifier = BatchClassifier(use_llm=False)  # 先不用 LLM
    
    # 测试数据
    test_data = [
        {"id": "1", "title": "国务院常务会议部署稳经济措施"},
        {"id": "2", "title": "央行宣布降准0.5个百分点 A股大涨"},
        {"id": "3", "title": "解放军东部战区组织海空联合演训"},
        {"id": "4", "title": "警方通报：男子持刀伤人致3死5伤"},
        {"id": "5", "title": "OpenAI发布GPT-5 性能大幅提升"},
        {"id": "6", "title": "梅西帽子戏法 阿根廷3-0大胜巴西"},
        {"id": "7", "title": "周杰伦新专辑首发 粉丝疯狂抢购"},
        {"id": "8", "title": "今日天气预报：多云转晴"},
        {"id": "9", "title": "比亚迪销量创新高 新能源车渗透率突破50%"},
        {"id": "10", "title": "特朗普宣布参加2024年大选"},
    ]
    
    print("=" * 60)
    print("批量分类器 演示")
    print("=" * 60)
    
    results = classifier.classify_batch(test_data, show_progress=False)
    
    print("\n分类结果:")
    for r in results:
        status = "⚠" if r.needs_review else "✓"
        print(f"  {status} [{r.category}] {r.title[:30]}...")
        if r.matched_keywords:
            print(f"       关键词: {r.matched_keywords[:3]}")
    
    classifier.print_stats()


if __name__ == '__main__':
    demo()
