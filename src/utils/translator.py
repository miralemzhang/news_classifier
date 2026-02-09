"""
翻译器模块
支持将关键词翻译为目标语言，用于多语言规则匹配
"""

from abc import ABC, abstractmethod
from typing import List
import json


class BaseTranslator(ABC):
    """翻译器基类"""

    @abstractmethod
    def translate(self, text: str, target_lang: str) -> str:
        """
        翻译单个文本

        Args:
            text: 要翻译的文本
            target_lang: 目标语言代码（如 'en', 'ja', 'ko', 'fr', 'de' 等）

        Returns:
            翻译后的文本
        """
        pass

    def translate_batch(self, texts: List[str], target_lang: str) -> List[str]:
        """
        批量翻译

        Args:
            texts: 要翻译的文本列表
            target_lang: 目标语言代码

        Returns:
            翻译后的文本列表
        """
        return [self.translate(text, target_lang) for text in texts]


class GoogleTranslator(BaseTranslator):
    """
    使用 Google Translate API 的翻译器

    需要安装: pip install googletrans==4.0.0-rc1
    """

    def __init__(self):
        try:
            from googletrans import Translator
            self.translator = Translator()
        except ImportError:
            raise ImportError(
                "请安装 googletrans: pip install googletrans==4.0.0-rc1"
            )

    def translate(self, text: str, target_lang: str) -> str:
        try:
            result = self.translator.translate(text, dest=target_lang, src='zh-cn')
            return result.text
        except Exception as e:
            print(f"翻译失败 '{text}': {e}")
            return text  # 返回原文

    def translate_batch(self, texts: List[str], target_lang: str) -> List[str]:
        """批量翻译（优化版本）"""
        if not texts:
            return []

        try:
            # googletrans 支持批量翻译
            results = self.translator.translate(texts, dest=target_lang, src='zh-cn')
            if isinstance(results, list):
                return [r.text for r in results]
            else:
                return [results.text]
        except Exception as e:
            print(f"批量翻译失败: {e}")
            # 回退到逐个翻译
            return [self.translate(text, target_lang) for text in texts]


class DeepLTranslator(BaseTranslator):
    """
    使用 DeepL API 的翻译器

    需要: DeepL API Key
    安装: pip install deepl
    """

    # 语言代码映射
    LANG_MAP = {
        'en': 'EN-US',
        'ja': 'JA',
        'ko': 'KO',
        'de': 'DE',
        'fr': 'FR',
        'es': 'ES',
        'it': 'IT',
        'pt': 'PT-BR',
        'ru': 'RU',
    }

    def __init__(self, api_key: str):
        try:
            import deepl
            self.translator = deepl.Translator(api_key)
        except ImportError:
            raise ImportError("请安装 deepl: pip install deepl")

        self.api_key = api_key

    def translate(self, text: str, target_lang: str) -> str:
        target = self.LANG_MAP.get(target_lang, target_lang.upper())
        try:
            result = self.translator.translate_text(text, target_lang=target)
            return result.text
        except Exception as e:
            print(f"翻译失败 '{text}': {e}")
            return text

    def translate_batch(self, texts: List[str], target_lang: str) -> List[str]:
        if not texts:
            return []

        target = self.LANG_MAP.get(target_lang, target_lang.upper())
        try:
            results = self.translator.translate_text(texts, target_lang=target)
            return [r.text for r in results]
        except Exception as e:
            print(f"批量翻译失败: {e}")
            return [self.translate(text, target_lang) for text in texts]


class CachedTranslator(BaseTranslator):
    """
    带缓存的翻译器包装器

    缓存翻译结果到文件，避免重复调用 API
    """

    def __init__(self, translator: BaseTranslator, cache_path: str = None):
        self.translator = translator
        self.cache_path = cache_path or "translation_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _save_cache(self):
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _cache_key(self, text: str, target_lang: str) -> str:
        return f"{target_lang}:{text}"

    def translate(self, text: str, target_lang: str) -> str:
        key = self._cache_key(text, target_lang)
        if key in self.cache:
            return self.cache[key]

        result = self.translator.translate(text, target_lang)
        self.cache[key] = result
        self._save_cache()
        return result

    def translate_batch(self, texts: List[str], target_lang: str) -> List[str]:
        # 分离已缓存和未缓存的
        results = {}
        to_translate = []
        to_translate_idx = []

        for i, text in enumerate(texts):
            key = self._cache_key(text, target_lang)
            if key in self.cache:
                results[i] = self.cache[key]
            else:
                to_translate.append(text)
                to_translate_idx.append(i)

        # 翻译未缓存的
        if to_translate:
            translated = self.translator.translate_batch(to_translate, target_lang)
            for i, (idx, text, result) in enumerate(
                zip(to_translate_idx, to_translate, translated)
            ):
                results[idx] = result
                key = self._cache_key(text, target_lang)
                self.cache[key] = result

            self._save_cache()

        # 按顺序返回
        return [results[i] for i in range(len(texts))]


def create_translator(translator_type: str = "google", **kwargs) -> BaseTranslator:
    """
    创建翻译器

    Args:
        translator_type: 翻译器类型 ('google', 'deepl')
        **kwargs: 其他参数（如 api_key）

    Returns:
        翻译器实例
    """
    if translator_type == "google":
        translator = GoogleTranslator()
    elif translator_type == "deepl":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("DeepL 翻译器需要 api_key")
        translator = DeepLTranslator(api_key)
    else:
        raise ValueError(f"不支持的翻译器类型: {translator_type}")

    # 包装缓存
    cache_path = kwargs.get("cache_path")
    if cache_path:
        translator = CachedTranslator(translator, cache_path)

    return translator


if __name__ == "__main__":
    # 测试
    print("测试翻译器...")

    try:
        translator = create_translator("google")

        test_keywords = ["央行", "降准", "股市", "人工智能"]
        print(f"\n原始关键词: {test_keywords}")

        for lang in ['en', 'ja', 'ko']:
            translated = translator.translate_batch(test_keywords, lang)
            print(f"{lang}: {translated}")

    except Exception as e:
        print(f"测试失败: {e}")
