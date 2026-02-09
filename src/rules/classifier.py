"""
规则分类模块
基于关键词和 URL 路径的规则分类器
支持多语言：中文(zh)、英文(en)、日文(ja)、韩文(ko)
其他语言通过动态翻译关键词进行匹配
"""

import json
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from urllib.parse import urlparse

# 支持的预翻译语言（联合国官方语言 + 日韩）
SUPPORTED_LANGUAGES = {'zh', 'en', 'ja', 'ko', 'ar', 'ru', 'fr', 'es'}

# 拉丁语系语言（使用拉丁字母，难以通过字符集区分）
LATIN_LANGUAGES = {'en', 'fr', 'es', 'de', 'pt', 'it'}


def detect_script(text: str) -> str:
    """
    检测文本的文字系统（script），返回对应的语言代码

    Returns:
        'zh' - 中文
        'ja' - 日文（含假名）
        'ko' - 韩文
        'ar' - 阿拉伯文
        'ru' - 俄文（西里尔字母）
        'latin' - 拉丁字母（英/法/德/西等，无法细分）
    """
    # 统计各类字符数量
    chinese_count = 0
    japanese_count = 0  # 平假名+片假名
    korean_count = 0
    arabic_count = 0
    cyrillic_count = 0
    latin_count = 0

    for char in text:
        code = ord(char)
        # 中文字符（CJK统一汉字）
        if 0x4E00 <= code <= 0x9FFF:
            chinese_count += 1
        # 日文假名
        elif (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF):
            japanese_count += 1
        # 韩文
        elif 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
            korean_count += 1
        # 阿拉伯文
        elif 0x0600 <= code <= 0x06FF:
            arabic_count += 1
        # 西里尔字母（俄文等）
        elif 0x0400 <= code <= 0x04FF:
            cyrillic_count += 1
        # 拉丁字母
        elif (0x0041 <= code <= 0x007A) or (0x00C0 <= code <= 0x00FF):
            latin_count += 1

    # 按数量判断主要语言
    counts = [
        ('ja', japanese_count),  # 日文优先（因为日文也用汉字）
        ('ko', korean_count),
        ('zh', chinese_count),
        ('ar', arabic_count),
        ('ru', cyrillic_count),
        ('latin', latin_count),
    ]

    # 过滤掉数量为0的
    counts = [(lang, cnt) for lang, cnt in counts if cnt > 0]

    if not counts:
        return 'latin'  # 默认

    # 返回数量最多的
    counts.sort(key=lambda x: -x[1])
    return counts[0][0]



@dataclass
class RuleResult:
    """规则分类结果"""
    label: str                      # 分类标签，如 "经济"
    confidence: float               # 置信度，0-1 之间，如 0.95
    evidence: str                   # 证据，如 "URL路径+关键词一致"
    matched_keywords: list[str]     # 匹配到的关键词，如 ["央行", "降准"]
    matched_url_pattern: str        # 匹配到的 URL 模式
    needs_llm: bool = False         # 是否需要 LLM 介入（置信度 < 0.80）
    candidate_labels: list[str] = None  # 候选类别（供 LLM 参考）

    '''
    图解：
    分类结果 RuleResult
    ┌─────────────────────────────────┐
    │ label = "经济"                   │
    │ confidence = 0.95               │
    │ evidence = "URL路径+关键词一致"   │
    │ matched_keywords = ["央行"]      │
    │ matched_url_pattern = "经济"     │
    │ needs_llm = False               │  ← 新增：是否需要LLM
    │ candidate_labels = ["经济"]     │  ← 新增：候选类别
    └─────────────────────────────────┘
    '''
    
# URL 路径特征
URL_PATH_PATTERNS = {
    "时政": [r"/politics/", r"/gov/", r"/policy/", r"/shizheng/", r"/guonei/", r"/guoji/"],
    "经济": [r"/finance/", r"/economy/", r"/stock/", r"/caijing/", r"/money/", r"/business/"],
    "军事": [r"/military/", r"/mil/", r"/junshi/", r"/war/", r"/defense/"],
    "社会": [r"/society/", r"/social/", r"/shehui/", r"/legal/", r"/law/"],
    "科技": [r"/tech/", r"/technology/", r"/keji/", r"/it/", r"/digital/", r"/science/"],
    "体育": [r"/sports/", r"/tiyu/", r"/nba/", r"/football/", r"/cba/", r"/esports/"],
    "娱乐": [r"/ent/", r"/entertainment/", r"/yule/", r"/star/", r"/movie/", r"/music/"]
}

'''
作用：通过 URL 路径快速判断类别

https://finance.sina.com.cn/stock/...
                            ↑
                        包含 /stock/
                            ↓
                        → 经济类
'''


class RuleClassifier:
    """规则分类器 - 支持多语言"""

    def __init__(self, keywords_path: str = None, translator=None):
        """
        初始化规则分类器

        Args:
            keywords_path: 中文关键词路径（默认 configs/keywords.json）
            translator: 翻译器实例，用于动态翻译（可选）
        """
        base_dir = Path(__file__).parent.parent.parent

        if keywords_path is None:
            keywords_path = base_dir / "configs" / "keywords.json"

        # 加载多语言关键词库
        self.keywords_by_lang = {}

        # 中文（默认）
        with open(keywords_path, 'r', encoding='utf-8') as f:
            self.keywords_by_lang['zh'] = json.load(f)

        # 英文
        en_path = base_dir / "configs" / "keywords_en.json"
        if en_path.exists():
            with open(en_path, 'r', encoding='utf-8') as f:
                self.keywords_by_lang['en'] = json.load(f)

        # 日文
        ja_path = base_dir / "configs" / "keywords_ja.json"
        if ja_path.exists():
            with open(ja_path, 'r', encoding='utf-8') as f:
                self.keywords_by_lang['ja'] = json.load(f)

        # 韩文
        ko_path = base_dir / "configs" / "keywords_ko.json"
        if ko_path.exists():
            with open(ko_path, 'r', encoding='utf-8') as f:
                self.keywords_by_lang['ko'] = json.load(f)

        # 阿拉伯文
        ar_path = base_dir / "configs" / "keywords_ar.json"
        if ar_path.exists():
            with open(ar_path, 'r', encoding='utf-8') as f:
                self.keywords_by_lang['ar'] = json.load(f)

        # 俄文
        ru_path = base_dir / "configs" / "keywords_ru.json"
        if ru_path.exists():
            with open(ru_path, 'r', encoding='utf-8') as f:
                self.keywords_by_lang['ru'] = json.load(f)

        # 法文
        fr_path = base_dir / "configs" / "keywords_fr.json"
        if fr_path.exists():
            with open(fr_path, 'r', encoding='utf-8') as f:
                self.keywords_by_lang['fr'] = json.load(f)

        # 西班牙文
        es_path = base_dir / "configs" / "keywords_es.json"
        if es_path.exists():
            with open(es_path, 'r', encoding='utf-8') as f:
                self.keywords_by_lang['es'] = json.load(f)

        # 保持向后兼容
        self.keywords = self.keywords_by_lang['zh']

        self.url_patterns = URL_PATH_PATTERNS
        # 8 个类别标签
        self.labels = ["时政", "经济", "军事", "社会", "科技", "体育", "娱乐", "其他"]

        # 优先级（数字越小优先级越高）
        self.priority = {
            "时政": 1, "军事": 2, "经济": 3, "社会": 4,
            "科技": 5, "体育": 6, "娱乐": 7, "其他": 8
        }

        # 翻译器（用于不支持的语言）
        self.translator = translator

        # 动态翻译缓存
        self._translated_keywords_cache = {}
        
        
        '''
        图解：                                                                            
        初始化时加载的数据                                                                
        ┌──────────────────────────────────────────┐                                      
        │ self.keywords (从 keywords.json 读取)     │                                     
        │ ┌──────────────────────────────────────┐ │                                      
        │ │ "时政": {"strong": ["国务院", "习近平"]}│ │                                   
        │ │ "经济": {"strong": ["央行", "降准"]}   │ │                                    
        │ │ ...                                   │ │                                     
        │ └──────────────────────────────────────┘ │                                      
        │                                          │                                      
        │ self.url_patterns                        │                                      
        │ ┌──────────────────────────────────────┐ │                                      
        │ │ "时政": ["/politics/", "/gov/"]       │ │                                     
        │ │ "经济": ["/finance/", "/stock/"]      │ │                                     
        │ │ ...                                   │ │                                     
        │ └──────────────────────────────────────┘ │                                      
        └──────────────────────────────────────────┘   
        '''
        
        
        
    
    def classify(self, title: str, content: str = "", url_path: str = "",
                 meta_keywords: str = "") -> Optional[RuleResult]:
        """
        使用规则分类（只使用匹配文本语言的关键词库）

        优先级：标题 > 正文（标题匹配权重更高）

        Args:
            title: 标题
            content: 内容
            url_path: URL 路径
            meta_keywords: 元关键词

        Returns:
            RuleResult: 分类结果
                - needs_llm=False: 置信度 >= 0.80，可直接使用
                - needs_llm=True:  置信度 < 0.80，建议 LLM 介入
                - None: 完全无法匹配，必须用 LLM
        """
        # 检测文本语言
        text_for_detect = f"{title} {content[:200]}"
        script = detect_script(text_for_detect)

        # 根据文字系统决定使用哪些关键词库
        if script == 'latin':
            # 拉丁字母：尝试所有拉丁语系关键词库（en, fr, es）
            langs_to_try = [lang for lang in self.keywords_by_lang.keys()
                           if lang in LATIN_LANGUAGES]
        elif script in self.keywords_by_lang:
            # 非拉丁字母且有对应关键词库：只用对应语言
            langs_to_try = [script]
        else:
            # 没有对应语言的关键词库（如德语）→ 交给 LLM
            return None

        # 尝试匹配的语言关键词库，取最高置信度结果
        best_result = None
        for lang in langs_to_try:
            result = self._classify_with_language(title, content, url_path, meta_keywords, lang)
            if result is not None:
                if best_result is None or result.confidence > best_result.confidence:
                    best_result = result
        return best_result

    def _classify_with_language(self, title: str, content: str, url_path: str,
                                meta_keywords: str, language: str) -> Optional[RuleResult]:
        """使用指定语言的关键词库进行分类"""
        # 获取对应语言的关键词库
        keywords = self._get_keywords_for_language(language)

        # 1. 检查 URL 路径（可能返回多个匹配）
        url_labels = self._match_url_pattern(url_path)

        # 2. 分别匹配标题和正文的强关键词
        title_strong = self._match_keywords(title, "strong", keywords)

        # 清理正文：去除推荐内容（play_circle_outline 后面通常是相关推荐）
        clean_content = content[:500]
        # 去除常见的推荐内容标记
        for marker in ['play_circle_outline', '相關報導', '相关报道', '延伸閱讀', '延伸阅读', '更多新聞', '更多新闻']:
            if marker in clean_content:
                clean_content = clean_content.split(marker)[0]

        content_text = f"{clean_content} {meta_keywords}"
        content_strong = self._match_keywords(content_text, "strong", keywords)

        # 3. 标题优先逻辑：如果标题有明确匹配，优先使用标题结果
        if title_strong:
            # 标题匹配到了，检查是否与正文冲突
            title_labels = set(title_strong.keys())
            content_labels = set(content_strong.keys()) if content_strong else set()

            # 如果标题和正文匹配不同类别，以标题为准
            if title_labels != content_labels and content_labels:
                # 只使用标题的匹配结果，忽略正文中的干扰
                strong_matches = title_strong
            else:
                # 标题和正文一致，或正文无匹配，合并结果
                strong_matches = self._merge_matches(title_strong, content_strong)
        else:
            # 标题无匹配，使用正文结果
            strong_matches = content_strong

        # 4. 检查弱关键词（标题+正文）
        full_text = f"{title} {content_text}"
        weak_matches = self._match_keywords(full_text, "weak", keywords)

        # 5. 综合判断
        return self._decide(url_labels, strong_matches, weak_matches)

    def _merge_matches(self, matches1: dict, matches2: dict) -> dict:
        """合并两个匹配结果字典"""
        merged = dict(matches1)
        for label, keywords in matches2.items():
            if label in merged:
                # 合并关键词列表，去重
                merged[label] = list(set(merged[label] + keywords))
            else:
                merged[label] = keywords
        return merged

    def _get_keywords_for_language(self, language: str) -> dict:
        """
        获取对应语言的关键词库

        支持的语言：zh, en, ja, ko
        其他语言：尝试动态翻译，否则回退到英文
        """
        # 如果是支持的语言，直接返回
        if language in self.keywords_by_lang:
            return self.keywords_by_lang[language]

        # 其他语言：尝试动态翻译
        if language in self._translated_keywords_cache:
            return self._translated_keywords_cache[language]

        # 如果有翻译器，动态翻译关键词
        if self.translator:
            try:
                translated = self._translate_keywords(language)
                self._translated_keywords_cache[language] = translated
                return translated
            except Exception as e:
                print(f"翻译关键词失败 ({language}): {e}")

        # 回退到英文（国际通用）
        return self.keywords_by_lang.get('en', self.keywords_by_lang['zh'])

    def _translate_keywords(self, target_lang: str) -> dict:
        """
        将中文关键词翻译为目标语言

        Args:
            target_lang: 目标语言代码

        Returns:
            翻译后的关键词字典
        """
        if not self.translator:
            raise ValueError("未配置翻译器")

        translated = {}
        zh_keywords = self.keywords_by_lang['zh']

        for label, kw_dict in zh_keywords.items():
            translated[label] = {}
            for kw_type in ['strong', 'weak']:
                original_keywords = kw_dict.get(kw_type, [])
                if original_keywords:
                    # 批量翻译
                    translated_kws = self.translator.translate_batch(
                        original_keywords, target_lang
                    )
                    translated[label][kw_type] = translated_kws
                else:
                    translated[label][kw_type] = []

        return translated
    
    
    
    '''
    流程图：                                                                          
    输入: title="央行宣布降准", url="https://finance.sina.com.cn/finance/..."                 
                                                                                        
            ┌─────────────────────────────────────────────┐                             
            │ text = "央行宣布降准 ..."                     │                           
            └─────────────────────────────────────────────┘                             
                                │                                                      
            ┌─────────────────┼─────────────────┐                                    
            ▼                 ▼                 ▼                                    
        ┌──────────┐     ┌──────────────┐   ┌──────────────┐                           
        │ URL 匹配  │     │ 强关键词匹配  │   │ 弱关键词匹配  │                        
        │ /finance/ │     │ "央行","降准" │   │   (无)       │                         
        │  → 经济   │     │   → 经济     │   │              │                          
        └──────────┘     └──────────────┘   └──────────────┘                           
            │                 │                 │                                    
            └─────────────────┼─────────────────┘                                    
                                ▼                                                      
                        ┌─────────────┐                                               
                        │  综合判断    │                                              
                        │ URL+关键词   │                                              
                        │ 都是经济     │                                              
                        │ 置信度=0.95  │                                              
                        └─────────────┘                                               
                                │                                                      
                                ▼                                                      
                        输出: 经济 (95%)                                                
                                            
    '''
    
    
    
    # def _match_url_pattern(self, url_path: str) -> Optional[str]:
    #     if not url_path:
    #         return None
    #     for label, patterns in self.url_patterns.items():
    #         for pattern in patterns:
    #             # 用正则表达式匹配（忽略大小写）                                      
    #             if re.search(pattern, url_path, re.IGNORECASE):
    #                 return label
    #     return None
    
    def _match_url_pattern(self, url_path: str) -> list[str]:
        """
        匹配 URL 路径，返回所有匹配的类别（可能有多个）

        Returns:
            list[str]: 匹配到的类别列表，如 ["经济", "科技"]
        """
        if not url_path:
            return []

        # 如果传进来的是完整 URL，就只取 path
        try:
            parsed = urlparse(url_path)
            if parsed.scheme and parsed.netloc:
                url_path = parsed.path
        except Exception:
            pass

        matched_labels = []
        for label, patterns in self.url_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url_path, re.IGNORECASE):
                    matched_labels.append(label)
                    break  # 一个类别只加一次

        return matched_labels

    '''
    例子：
    URL = "https://finance.sina.com.cn/tech/stock/..."

    可能同时匹配:
      - /finance/ → 经济
      - /tech/    → 科技
      - /stock/   → 经济（已加过，不重复）

    返回: ["经济", "科技"]
    '''

    def _match_keywords(self, text: str, keyword_type: str,
                        keywords: dict = None) -> dict[str, list[str]]:
        """
        匹配关键词（使用词边界匹配英文，避免 ceasefire 匹配 fire）

        Args:
            text: 要匹配的文本
            keyword_type: 关键词类型 ('strong' 或 'weak')
            keywords: 关键词字典（可选，默认使用中文关键词）
        """
        if keywords is None:
            keywords = self.keywords

        matches = {}  # 结果字典
        text_lower = text.lower()

        # 遍历每个类别的关键词
        for label, kw_dict in keywords.items():
            if label == "其他":
                continue
            matched = []
            for kw in kw_dict.get(keyword_type, []):
                # 判断是否为纯英文关键词（使用词边界匹配）
                if re.match(r'^[a-zA-Z\s\-\']+$', kw):
                    # 英文关键词：使用词边界匹配，避免 ceasefire 匹配 fire
                    pattern = r'\b' + re.escape(kw.lower()) + r'\b'
                    if re.search(pattern, text_lower):
                        matched.append(kw)
                else:
                    # 非英文关键词（中文等）：直接包含匹配
                    if kw in text:
                        matched.append(kw)
            if matched:
                matches[label] = matched
        return matches
    
    """
    例子：                                                                            
    输入: text = "央行宣布降准0.5个百分点"                                            
            keyword_type = "strong"                                                     
                                                                                        
    关键词库:                                                                         
        经济.strong = ["央行", "降准", "GDP", ...]                                      
        时政.strong = ["国务院", "习近平", ...]                                         
                                                                                        
    检查:                                                                             
        "央行" in text? ✅                                                              
        "降准" in text? ✅                                                              
        "GDP" in text? ❌                                                               
        "国务院" in text? ❌                                                            
                                                                                        
    返回: {"经济": ["央行", "降准"]}  
    """
    
    def _decide(self, url_labels: list[str], strong_matches: dict,
                weak_matches: dict) -> Optional[RuleResult]:
        """
        综合判断分类结果

        新规则：
        - 置信度 >= 0.80 → 直接输出分类
        - 置信度 < 0.80 → 需要 LLM 介入（needs_llm=True）

        Args:
            url_labels: URL 匹配到的类别列表（可能多个）
            strong_matches: 强关键词匹配结果
            weak_matches: 弱关键词匹配结果
        """

        # 收集所有候选类别
        all_candidates = set(url_labels) | set(strong_matches.keys()) | set(weak_matches.keys())
        if not all_candidates:
            all_candidates = {"其他"}
        candidate_list = list(all_candidates)

        # ═══════════════════════════════════════════════════════════════
        # 规则1：URL + 强关键词 一致 → 高置信度 0.95
        # ═══════════════════════════════════════════════════════════════
        for url_label in url_labels:
            if url_label in strong_matches:
                return RuleResult(
                    label=url_label,
                    confidence=0.95,
                    evidence="URL路径+关键词一致",
                    matched_keywords=strong_matches[url_label],
                    matched_url_pattern=url_label,
                    needs_llm=False,  # >= 0.80，不需要 LLM
                    candidate_labels=candidate_list
                )

        # ═══════════════════════════════════════════════════════════════
        # 规则2：有强关键词匹配 → 根据数量决定置信度
        # ═══════════════════════════════════════════════════════════════
        if strong_matches:
            # 按优先级排序
            sorted_labels = sorted(strong_matches.keys(),
                                   key=lambda x: self.priority.get(x, 99))
            best_label = sorted_labels[0]
            count = len(strong_matches[best_label])

            # 计算置信度
            if count >= 3:
                conf = 0.90
            elif count >= 2:
                conf = 0.85
            else:
                conf = 0.75  # 只有1个强关键词

            return RuleResult(
                label=best_label,
                confidence=conf,
                evidence=f"匹配{count}个强关键词",
                matched_keywords=strong_matches[best_label][:5],
                matched_url_pattern=url_labels[0] if url_labels else "",
                needs_llm=(conf < 0.80),  # < 0.80 需要 LLM
                candidate_labels=candidate_list
            )

        # ═══════════════════════════════════════════════════════════════
        # 规则3：仅 URL 匹配 → 低置信度，需要 LLM
        # ═══════════════════════════════════════════════════════════════
        if url_labels:
            # 如果 URL 匹配了多个类别，按优先级选
            best_url_label = min(url_labels, key=lambda x: self.priority.get(x, 99))
            return RuleResult(
                label=best_url_label,
                confidence=0.50,
                evidence=f"仅URL路径匹配 (共{len(url_labels)}个候选)",
                matched_keywords=[],
                matched_url_pattern=best_url_label,
                needs_llm=True,  # < 0.80，需要 LLM
                candidate_labels=candidate_list
            )

        # ═══════════════════════════════════════════════════════════════
        # 规则4：仅弱关键词 → 需要 LLM
        # ═══════════════════════════════════════════════════════════════
        if weak_matches:
            sorted_labels = sorted(weak_matches.keys(),
                                   key=lambda x: self.priority.get(x, 99))
            best_label = sorted_labels[0]
            return RuleResult(
                label=best_label,
                confidence=0.40,
                evidence="仅匹配弱关键词",
                matched_keywords=weak_matches[best_label][:3],
                matched_url_pattern="",
                needs_llm=True,  # < 0.80，需要 LLM
                candidate_labels=candidate_list
            )

        # ═══════════════════════════════════════════════════════════════
        # 规则5：什么都没匹配 → 返回 None，完全交给 LLM
        # ═══════════════════════════════════════════════════════════════
        return None
    
    '''
      决策流程图：                                                                      
                      ┌─────────────────┐                                           
                      │   开始决策       │                                          
                      └────────┬────────┘                                           
                               │                                                    
                               ▼                                                    
                ┌──────────────────────────────┐                                    
                │ URL + 强关键词 一致？          │                                  
                └──────────────┬───────────────┘                                    
                      YES │          │ NO                                           
                          ▼          ▼                                              
                ┌─────────────┐    ┌──────────────────────┐                         
                │ 返回 95%    │    │ 有强关键词匹配？       │                       
                │ 置信度      │    └──────────┬───────────┘                         
                └─────────────┘      YES │          │ NO                            
                                         ▼          ▼                               
                               ┌─────────────┐   ┌───────────────┐                 
                               │ 返回 70-90%  │    │ 只有URL匹配？  │                
                               │ 置信度       │    └───────┬───────┘                 
                               └─────────────┘   YES │        │ NO                  
                                                     ▼        ▼                     
                                            ┌─────────┐  ┌─────────┐                
                                            │返回 50% │   │返回 None│                
                                            │置信度   │   │交给 LLM │                
                                            └─────────┘  └─────────┘   
    '''
    
    
    def get_candidate_labels(self, title: str, content: str = "",
                             url_path: str = "", top_k: int = 3) -> list[str]:
        """获取候选类别（用于约束 LLM）"""
        text = f"{title} {content[:500]}"
        scores = {label: 0 for label in self.labels}

        # URL 加分（现在可能有多个匹配）
        url_labels = self._match_url_pattern(url_path)
        for url_label in url_labels:
            scores[url_label] += 5

        # 关键词加分
        for label, kw_dict in self.keywords.items():
            for kw in kw_dict.get("strong", []):
                if kw in text:
                    scores[label] += 3
            for kw in kw_dict.get("weak", []):
                if kw in text:
                    scores[label] += 1

        # 排序取 Top-K
        sorted_labels = sorted(scores.items(), key=lambda x: -x[1])
        top = [label for label, score in sorted_labels[:top_k] if score > 0]

        if not top:
            return self.labels
        if "其他" not in top:
            top.append("其他")
        return top
    
    def get_stats(self) -> dict:
        """获取关键词统计"""
        
        # 初始化计数器
        stats = {}
        total_strong = 0
        total_weak = 0
        
        # 遍历每个类别，数 strong/weak 数量
        for label, kw_dict in self.keywords.items():
            strong_count = len(kw_dict.get("strong", []))
            weak_count = len(kw_dict.get("weak", []))
            stats[label] = {"strong": strong_count, "weak": weak_count}
            total_strong += strong_count
            total_weak += weak_count
        # 总计
        stats["_total"] = {"strong": total_strong, "weak": total_weak}
        return stats
    
    '''
        最终输出像这样：

        {
        "时政": {"strong": 120, "weak": 300},
        "经济": {"strong": 90,  "weak": 260},
        ...
        "_total": {"strong": 600, "weak": 1600}
        }
    '''


if __name__ == '__main__':
    classifier = RuleClassifier()

    # 打印关键词统计
    print("=== 关键词统计 ===")
    stats = classifier.get_stats()
    for label, counts in stats.items():
        if label != "_total":
            print(f"{label}: 强={counts['strong']}, 弱={counts['weak']}")
    print(f"总计: 强={stats['_total']['strong']}, 弱={stats['_total']['weak']}")

    # 测试
    print("\n=== 测试分类（新规则：置信度 >= 0.80 才直接输出）===")
    test_cases = [
        {"title": "国务院常务会议部署稳经济措施", "url_path": "/politics/"},
        {"title": "央行宣布降准0.5个百分点", "url_path": "/finance/"},
        {"title": "解放军东部战区海空联合演训", "url_path": "/mil/"},
        {"title": "OpenAI发布GPT-5模型", "url_path": "/tech/"},
        {"title": "梅西加盟迈阿密国际", "url_path": "/sports/"},
        {"title": "周杰伦新专辑首发", "url_path": "/ent/"},
        # 新增测试：低置信度情况
        {"title": "今日天气预报", "url_path": "/weather/"},
        {"title": "某公司发布财报", "url_path": ""},
    ]

    print("-" * 70)
    for case in test_cases:
        result = classifier.classify(title=case["title"], url_path=case.get("url_path", ""))
        print(f"\n标题: {case['title']}")
        print(f"URL:  {case.get('url_path', '无')}")

        if result:
            status = "✅ 直接输出" if not result.needs_llm else "⚠️  需要LLM"
            print(f"  → {result.label} (置信度: {result.confidence:.0%}) {status}")
            print(f"     证据: {result.evidence}")
            if result.matched_keywords:
                print(f"     关键词: {result.matched_keywords[:3]}")
            if result.candidate_labels:
                print(f"     候选类别: {result.candidate_labels}")
        else:
            print(f"  → 无法分类，完全交给 LLM")

    print("\n" + "=" * 70)
    print("规则说明:")
    print("  置信度 >= 0.80 → 直接使用规则分类结果")
    print("  置信度 < 0.80  → 需要 LLM 介入确认")
    print("  无法匹配      → 完全交给 LLM 处理")
    print("=" * 70)
