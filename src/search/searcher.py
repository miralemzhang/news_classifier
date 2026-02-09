"""
智能搜索模块
根据用户查询自动定位分类，使用 LLM 读取内容，返回 Top K 相关结果

流程：
1. 用户输入查询 → 如 "2020 中国 失业人口"
2. 自动定位分类 → 根据关键词匹配到 "时政" 类别
3. 在该分类的数据中搜索相关文档
4. 使用 LLM 提取/总结相关信息
5. 返回 Top K 结果给用户
"""

import json
import os
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.classifier import RuleClassifier


@dataclass
class SearchResult:
    """单条搜索结果"""
    rank: int                    # 排名
    title: str                   # 标题
    url: str                     # URL
    category: str                # 分类
    snippet: str                 # 摘要片段
    relevance_score: float       # 相关性分数
    source_file: str             # 来源文件
    matched_keywords: list = field(default_factory=list)  # 匹配的关键词


@dataclass 
class SearchResponse:
    """搜索响应"""
    query: str                   # 原始查询
    detected_category: str       # 检测到的分类
    total_found: int             # 找到的总数
    results: list                # 结果列表 (Top K)
    llm_summary: str = ""        # LLM 总结（可选）


class SmartSearcher:
    """智能搜索器"""
    
    def __init__(
        self,
        data_dir: str = None,
        keywords_path: str = None,
        use_llm: bool = True,
        llm_model: str = "qwen2:7b",
        top_k: int = 15
    ):
        """
        初始化搜索器
        
        Args:
            data_dir: 数据目录（包含分类后的网页数据）
            keywords_path: 关键词库路径
            use_llm: 是否使用 LLM 进行内容提取和总结
            llm_model: LLM 模型名称
            top_k: 返回结果数量
        """
        base_dir = Path(__file__).parent.parent.parent
        
        if data_dir is None:
            self.data_dir = base_dir / "data" / "classified"
        else:
            self.data_dir = Path(data_dir)
        
        if keywords_path is None:
            keywords_path = base_dir / "configs" / "keywords.json"
        
        # 加载关键词库
        with open(keywords_path, 'r', encoding='utf-8') as f:
            self.keywords = json.load(f)
        
        self.rule_classifier = RuleClassifier(str(keywords_path))
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.top_k = top_k
        
        # LLM 客户端（延迟初始化）
        self._llm_client = None
        
        # 所有类别
        self.categories = ["时政", "经济", "军事", "社会", "科技", "体育", "娱乐", "其他"]
    
    @property
    def llm_client(self):
        """延迟加载 LLM 客户端"""
        if self._llm_client is None and self.use_llm:
            try:
                import ollama
                self._llm_client = ollama.Client()
            except ImportError:
                print("警告: ollama 未安装，将禁用 LLM 功能")
                self.use_llm = False
        return self._llm_client
    
    # 额外的领域关键词映射（补充关键词库中没有的常见查询词）
    DOMAIN_KEYWORDS = {
        "时政": ["失业", "就业", "民生", "人口", "统计", "政策", "扶贫", "脱贫", "社保", "养老", 
                 "医保", "教育", "住房", "民政", "救助", "低保", "疫情防控", "政府工作报告"],
        "经济": ["GDP", "增长", "股市", "房价", "通胀", "利率", "汇率", "投资", "消费", "贸易",
                 "进出口", "财政", "税收", "金融", "银行", "保险", "证券"],
        "军事": ["军队", "国防", "演习", "武器", "战争", "冲突", "军事"],
        "社会": ["事故", "犯罪", "灾害", "救援", "治安", "公益", "慈善"],
        "科技": ["AI", "人工智能", "5G", "芯片", "科学", "技术", "创新", "研发"],
        "体育": ["比赛", "冠军", "世界杯", "奥运", "球员", "联赛"],
        "娱乐": ["明星", "电影", "电视剧", "综艺", "歌手", "演员"]
    }
    
    def detect_category(self, query: str) -> tuple[str, list[str], float]:
        """
        根据查询词检测分类
        
        Returns:
            (category, matched_keywords, confidence)
        """
        scores = {cat: 0 for cat in self.categories}
        matched_kws = {cat: [] for cat in self.categories}
        
        query_lower = query.lower()
        
        # 1. 首先检查领域关键词映射
        for category, domain_kws in self.DOMAIN_KEYWORDS.items():
            for kw in domain_kws:
                if kw.lower() in query_lower or kw in query:
                    scores[category] += 2
                    matched_kws[category].append(kw)
        
        # 2. 然后检查关键词库
        for category, kw_dict in self.keywords.items():
            if category == "其他":
                continue
            
            # 强关键词匹配 (+3 分)
            for kw in kw_dict.get("strong", []):
                if kw.lower() in query_lower or kw in query:
                    scores[category] += 3
                    matched_kws[category].append(kw)
            
            # 弱关键词匹配 (+1 分)
            for kw in kw_dict.get("weak", []):
                if kw.lower() in query_lower or kw in query:
                    scores[category] += 1
                    matched_kws[category].append(kw)
        
        # 找出最高分的类别
        best_cat = max(scores, key=scores.get)
        best_score = scores[best_cat]
        
        if best_score == 0:
            return None, [], 0.0  # 返回 None 表示需要跨分类搜索
        
        # 计算置信度
        total_score = sum(scores.values())
        confidence = best_score / total_score if total_score > 0 else 0
        
        return best_cat, matched_kws[best_cat], confidence
    
    def search(self, query: str, category: str = None, top_k: int = None) -> SearchResponse:
        """
        执行搜索
        
        Args:
            query: 用户查询
            category: 指定类别（可选，若不指定则自动检测）
            top_k: 返回数量（可选）
        
        Returns:
            SearchResponse 对象
        """
        if top_k is None:
            top_k = self.top_k
        
        # 1. 检测分类
        cross_category_search = False
        if category is None:
            detected_cat, matched_kws, confidence = self.detect_category(query)
            print(f"🔍 查询: {query}")
            
            if detected_cat is None or confidence < 0.3:
                print(f"📂 无法确定分类，将跨所有分类搜索")
                cross_category_search = True
                detected_cat = "全部"
            else:
                print(f"📂 自动定位分类: {detected_cat} (置信度: {confidence:.2%})")
                if matched_kws:
                    print(f"🔑 匹配关键词: {', '.join(matched_kws[:5])}")
        else:
            detected_cat = category
            matched_kws = []
            print(f"🔍 查询: {query}")
            print(f"📂 指定分类: {detected_cat}")
        
        # 2. 在该分类下搜索文档
        results = self._search_in_category(query, detected_cat, top_k * 2)  # 多取一些用于排序
        
        # 3. 使用 LLM 增强排序和提取（可选）
        if self.use_llm and results:
            results = self._llm_rerank_and_extract(query, results, top_k)
        
        # 4. 取 Top K
        results = results[:top_k]
        
        # 5. 添加排名
        for i, r in enumerate(results):
            r.rank = i + 1
        
        # 6. 生成总结（可选）
        llm_summary = ""
        if self.use_llm and results:
            llm_summary = self._generate_summary(query, results)
        
        return SearchResponse(
            query=query,
            detected_category=detected_cat,
            total_found=len(results),
            results=results,
            llm_summary=llm_summary
        )
    
    def _search_in_category(self, query: str, category: str, limit: int) -> list[SearchResult]:
        """在指定分类下搜索文档，如果 category='全部' 则跨所有分类搜索"""
        results = []
        
        # 分词（简单按空格分割）
        query_terms = query.split()
        
        # 确定要搜索的分类
        if category == "全部":
            categories_to_search = self.categories
        else:
            categories_to_search = [category]
        
        for cat in categories_to_search:
            # 数据目录
            cat_dir = self.data_dir / cat
            
            if not cat_dir.exists():
                continue
            
            # 遍历该分类下的所有文件
            for file_path in cat_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        doc = json.load(f)
                    
                    # 计算相关性分数
                    score, matched = self._calculate_relevance(doc, query_terms)
                    
                    if score > 0:
                        results.append(SearchResult(
                            rank=0,
                            title=doc.get("title", "无标题"),
                            url=doc.get("url", ""),
                            category=cat,
                            snippet=self._extract_snippet(doc, query_terms),
                            relevance_score=score,
                            source_file=str(file_path),
                            matched_keywords=matched
                        ))
                except Exception as e:
                    continue
        
        if not results and category != "全部":
            print(f"⚠️ 分类 '{category}' 下未找到匹配文档")
        
        # 按相关性排序
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results[:limit]
    
    def _calculate_relevance(self, doc: dict, query_terms: list) -> tuple[float, list]:
        """计算文档相关性分数"""
        score = 0.0
        matched = []
        
        # 合并文档文本
        title = doc.get("title", "")
        content = doc.get("content", "")
        meta = doc.get("meta_keywords", "") + " " + doc.get("meta_description", "")
        
        text = f"{title} {content} {meta}".lower()
        
        for term in query_terms:
            term_lower = term.lower()
            
            # 标题匹配权重更高
            if term_lower in title.lower():
                score += 5
                matched.append(term)
            # 正文匹配
            elif term_lower in text:
                score += 1
                matched.append(term)
        
        # 所有词都匹配加成
        if len(matched) == len(query_terms) and len(query_terms) > 1:
            score *= 1.5
        
        return score, list(set(matched))
    
    def _extract_snippet(self, doc: dict, query_terms: list, max_len: int = 200) -> str:
        """提取包含查询词的摘要片段"""
        content = doc.get("content", "")
        
        if not content:
            return doc.get("meta_description", "")[:max_len]
        
        # 尝试找到包含查询词的段落
        for term in query_terms:
            pos = content.lower().find(term.lower())
            if pos != -1:
                start = max(0, pos - 50)
                end = min(len(content), pos + max_len - 50)
                snippet = content[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(content):
                    snippet = snippet + "..."
                return snippet
        
        # 没找到就返回开头
        return content[:max_len] + ("..." if len(content) > max_len else "")
    
    def _llm_rerank_and_extract(self, query: str, results: list, top_k: int) -> list:
        """使用 LLM 重新排序并提取关键信息"""
        if not self.llm_client:
            return results
        
        print(f"\n🤖 使用 LLM ({self.llm_model}) 进行智能排序和信息提取...")
        
        # 对 top 结果进行 LLM 评估
        for result in results[:top_k]:
            try:
                prompt = f"""请评估以下文档与查询的相关性，并提取关键信息。

查询: {query}

文档标题: {result.title}
文档摘要: {result.snippet}

请返回 JSON 格式:
{{"relevance": 0.0-1.0, "key_info": "与查询相关的关键信息摘要"}}
"""
                response = self.llm_client.generate(
                    model=self.llm_model,
                    prompt=prompt,
                    options={"temperature": 0.1, "num_predict": 256}
                )
                
                # 解析响应
                output = response.get('response', '')
                json_match = re.search(r'\{[^{}]*\}', output)
                if json_match:
                    data = json.loads(json_match.group(0))
                    result.relevance_score *= float(data.get("relevance", 0.5))
                    if "key_info" in data:
                        result.snippet = data["key_info"]
            except Exception as e:
                continue
        
        # 重新排序
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
    
    def _generate_summary(self, query: str, results: list) -> str:
        """使用 LLM 生成搜索结果总结"""
        if not self.llm_client or not results:
            return ""
        
        # 准备上下文
        context = "\n\n".join([
            f"[{r.rank}] {r.title}\n{r.snippet}"
            for r in results[:5]  # 取 top 5 生成总结
        ])
        
        prompt = f"""基于以下搜索结果，回答用户的查询问题。

用户查询: {query}

相关文档:
{context}

请用简洁的中文总结回答用户的问题，如果信息不足请说明:
"""
        
        try:
            response = self.llm_client.generate(
                model=self.llm_model,
                prompt=prompt,
                options={"temperature": 0.3, "num_predict": 512}
            )
            return response.get('response', '')
        except Exception as e:
            return f"生成总结时出错: {e}"


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="智能网页搜索")
    parser.add_argument("query", nargs="?", help="搜索查询")
    parser.add_argument("-c", "--category", help="指定分类")
    parser.add_argument("-k", "--top-k", type=int, default=15, help="返回结果数量")
    parser.add_argument("--no-llm", action="store_true", help="禁用 LLM")
    parser.add_argument("-d", "--data-dir", help="数据目录")
    parser.add_argument("-m", "--model", default="qwen2:7b", help="LLM 模型")
    
    args = parser.parse_args()
    
    # 创建搜索器
    searcher = SmartSearcher(
        data_dir=args.data_dir,
        use_llm=not args.no_llm,
        llm_model=args.model,
        top_k=args.top_k
    )
    
    # 交互模式
    if not args.query:
        print("=" * 60)
        print("🔍 智能网页搜索系统")
        print("=" * 60)
        print("输入查询内容，系统会自动定位分类并返回相关结果")
        print("输入 'quit' 或 'exit' 退出\n")
        
        while True:
            try:
                query = input("🔍 请输入查询: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    print("再见！")
                    break
                if not query:
                    continue
                
                response = searcher.search(query, category=args.category)
                _print_response(response)
                
            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
    else:
        response = searcher.search(args.query, category=args.category)
        _print_response(response)


def _print_response(response: SearchResponse):
    """打印搜索响应"""
    print("\n" + "=" * 60)
    print(f"📊 搜索结果 (共找到 {response.total_found} 条)")
    print("=" * 60)
    
    if not response.results:
        print("❌ 未找到相关结果")
        return
    
    for r in response.results:
        print(f"\n【{r.rank}】{r.title}")
        print(f"    📂 分类: {r.category}")
        print(f"    🔗 URL: {r.url}")
        print(f"    📊 相关性: {r.relevance_score:.2f}")
        print(f"    📝 {r.snippet[:100]}...")
    
    if response.llm_summary:
        print("\n" + "=" * 60)
        print("🤖 AI 总结:")
        print("=" * 60)
        print(response.llm_summary)


if __name__ == "__main__":
    main()
