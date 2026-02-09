"""
标题查询模块
支持分词匹配标题中的关键词
"""

import json
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# 尝试导入 jieba 分词
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("警告: jieba 未安装，将使用简单分词")


@dataclass
class QueryResult:
    """查询结果"""
    title: str
    matched_words: list[str]
    match_count: int
    match_ratio: float
    category: str = ""
    confidence: float = 0.0


class TitleMatcher:
    """标题匹配器"""
    
    def __init__(self, keywords_path: str = None):
        if keywords_path is None:
            base_dir = Path(__file__).parent.parent.parent
            keywords_path = base_dir / "configs" / "keywords.json"
        
        with open(keywords_path, 'r', encoding='utf-8') as f:
            self.keywords = json.load(f)
        
        # 构建所有关键词的集合（用于快速查找）
        self.all_keywords = set()
        self.keyword_to_category = {}
        for category, kw_dict in self.keywords.items():
            for kw in kw_dict.get("strong", []):
                self.all_keywords.add(kw)
                self.keyword_to_category[kw] = category
            for kw in kw_dict.get("weak", []):
                self.all_keywords.add(kw)
                if kw not in self.keyword_to_category:
                    self.keyword_to_category[kw] = category
        
        # 初始化 jieba
        if JIEBA_AVAILABLE:
            # 将关键词添加到 jieba 词典
            for kw in self.all_keywords:
                jieba.add_word(kw)
    
    def tokenize(self, text: str) -> list[str]:
        """
        分词
        支持中英文混合
        """
        if JIEBA_AVAILABLE:
            # 使用 jieba 分词
            words = list(jieba.cut(text))
        else:
            # 简单分词：按空格、标点分割
            words = re.split(r'[\s\u3000\uff0c\u3001\uff1a\uff1b\uff01\uff1f\u3002\uff08\uff09\u300a\u300b\u3010\u3011,.:;!?\(\)\[\]]+', text)
        
        # 过滤空字符串和单字符
        words = [w.strip() for w in words if len(w.strip()) >= 1]
        return words
    
    def extract_query_keywords(self, query: str) -> list[str]:
        """
        从查询中提取关键词
        过滤掉停用词和数字
        """
        # 停用词
        stopwords = {'的', '是', '在', '了', '和', '与', '或', '对', '为', '年', '月', '日', '时', '分', '秒',
                     '个', '位', '名', '次', '等', '这', '那', '有', '到', '从', '被', '把', '给', '让', '使',
                     '着', '过', '地', '得', '也', '都', '就', '只', '又', '但', '而', '以', '及', '其'}
        
        words = self.tokenize(query)
        
        # 过滤停用词和纯数字
        keywords = []
        for w in words:
            if w not in stopwords and not w.isdigit() and len(w) >= 2:
                keywords.append(w)
            elif w.isdigit() and len(w) == 4:  # 保留年份
                keywords.append(w)
        
        return keywords
    
    def match_title(self, query: str, title: str) -> QueryResult:
        """
        匹配查询词和标题
        返回匹配结果
        """
        query_keywords = self.extract_query_keywords(query)
        title_lower = title.lower()
        
        matched = []
        for kw in query_keywords:
            if kw.lower() in title_lower or kw in title:
                matched.append(kw)
        
        match_ratio = len(matched) / len(query_keywords) if query_keywords else 0
        
        return QueryResult(
            title=title,
            matched_words=matched,
            match_count=len(matched),
            match_ratio=match_ratio
        )
    
    def search_by_keywords(self, query: str, titles: list[str], min_match: int = 1) -> list[QueryResult]:
        """
        在标题列表中搜索匹配查询词的标题
        
        Args:
            query: 查询字符串（如 "中国2020 社会人口"）
            titles: 标题列表
            min_match: 最少匹配词数
        
        Returns:
            匹配结果列表，按匹配度排序
        """
        results = []
        
        for title in titles:
            result = self.match_title(query, title)
            if result.match_count >= min_match:
                results.append(result)
        
        # 按匹配数和匹配率排序
        results.sort(key=lambda x: (x.match_count, x.match_ratio), reverse=True)
        return results
    
    def extract_title_keywords(self, title: str) -> dict:
        """
        从标题中提取所有匹配的分类关键词
        返回每个类别匹配到的关键词
        """
        result = {}
        
        for category, kw_dict in self.keywords.items():
            matched_strong = []
            matched_weak = []
            
            for kw in kw_dict.get("strong", []):
                if kw in title:
                    matched_strong.append(kw)
            
            for kw in kw_dict.get("weak", []):
                if kw in title:
                    matched_weak.append(kw)
            
            if matched_strong or matched_weak:
                result[category] = {
                    "strong": matched_strong,
                    "weak": matched_weak,
                    "total": len(matched_strong) * 3 + len(matched_weak)  # 加权分数
                }
        
        return result
    
    def classify_title(self, title: str) -> tuple[str, float, list[str]]:
        """
        根据关键词分类标题
        
        Returns:
            (类别, 置信度, 匹配的关键词)
        """
        matches = self.extract_title_keywords(title)
        
        if not matches:
            return ("其他", 0.3, [])
        
        # 按加权分数排序
        sorted_cats = sorted(matches.items(), key=lambda x: x[1]["total"], reverse=True)
        best_cat = sorted_cats[0][0]
        best_match = sorted_cats[0][1]
        
        # 计算置信度
        if best_match["strong"]:
            confidence = min(0.7 + len(best_match["strong"]) * 0.1, 0.95)
        else:
            confidence = min(0.4 + len(best_match["weak"]) * 0.1, 0.6)
        
        all_matched = best_match["strong"] + best_match["weak"]
        return (best_cat, confidence, all_matched)


def demo():
    """演示功能"""
    matcher = TitleMatcher()
    
    print("=" * 60)
    print("标题关键词匹配器 演示")
    print("=" * 60)
    
    # 测试分词
    print("\n【1. 分词测试】")
    test_queries = [
        "中国2020 社会人口",
        "美国总统拜登访华",
        "A股暴跌 券商股跌停",
        "解放军东部战区军演"
    ]
    
    for query in test_queries:
        keywords = matcher.extract_query_keywords(query)
        print(f"  查询: {query}")
        print(f"  分词: {keywords}")
        print()
    
    # 测试标题分类
    print("\n【2. 标题分类测试】")
    test_titles = [
        "国务院常务会议部署稳经济措施",
        "央行宣布降准0.5个百分点 A股大涨",
        "解放军东部战区组织海空联合演训",
        "警方通报：男子持刀伤人致3死5伤",
        "OpenAI发布GPT-5 性能大幅提升",
        "梅西帽子戏法 阿根廷3-0大胜",
        "周杰伦新专辑首发 粉丝疯狂抢购",
        "今日天气：多云转晴 气温回升"
    ]
    
    for title in test_titles:
        category, confidence, matched = matcher.classify_title(title)
        print(f"  标题: {title}")
        print(f"  分类: {category} (置信度: {confidence:.2f})")
        print(f"  关键词: {matched[:5]}")
        print()
    
    # 测试查询匹配
    print("\n【3. 查询匹配测试】")
    query = "中国 军事 演习"
    print(f"  查询: {query}")
    
    sample_titles = [
        "解放军东部战区组织海空联合演训",
        "中国海军在南海进行实弹演习",
        "美军航母驶入南海 中方回应",
        "中国经济增长超预期",
        "军事专家解读中美关系"
    ]
    
    results = matcher.search_by_keywords(query, sample_titles, min_match=1)
    print(f"  匹配结果:")
    for r in results:
        print(f"    [{r.match_count}词] {r.title}")
        print(f"           匹配: {r.matched_words}")


if __name__ == '__main__':
    demo()
