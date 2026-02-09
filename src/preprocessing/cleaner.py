#!/usr/bin/env python3
"""
数据清洗模块
处理原始抓取数据，生成干净的分类输入

功能：
1. 清理文本中的噪音（换行符、HTML残留、JS/CSS代码等）
2. 去重
3. 乱码检测

使用方法：
    # 命令行
    python -m src.preprocessing.cleaner input.json -o output.jsonl

    # 代码中
    from src.preprocessing.cleaner import clean_raw_data, DataCleaner
    clean_raw_data("input.json", "output.jsonl")
"""

import re
import json
import html
import hashlib
import argparse
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict


@dataclass
class CleanedItem:
    """清洗后的数据项"""
    id: str
    title: str
    url: str
    content: str
    publish_time: str


class DataCleaner:
    """数据清洗器"""

    def __init__(self):
        # 统计信息
        self.stats = {
            "total": 0,
            "cleaned": 0,
            "duplicates": 0,
            "empty": 0,
            "garbled": 0,
        }

    def clean_text(self, text: str) -> str:
        """
        清理文本
        - 移除单字符换行（如 "大\n紀\n元" -> "大紀元"）
        - 移除HTML标签残留
        - 移除JavaScript/CSS代码
        - 移除多余空白
        """
        if not text:
            return ""

        # 1. 检测并修复单字符换行问题
        text = self._fix_char_by_char_newlines(text)

        # 2. 移除 JavaScript 代码块
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'var\s+\w+\s*=\s*[\[\{].*?[\]\}];?', '', text, flags=re.DOTALL)
        text = re.sub(r'function\s*\([^)]*\)\s*\{.*?\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\b(document|window|localStorage|getElementById|querySelector)\b[^;]*;?', '', text)

        # 2.5 移除 CSS 代码块
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)  # CSS 注释
        text = re.sub(r'@media[^{]*\{[^}]*\}', '', text, flags=re.DOTALL)  # @media queries
        text = re.sub(r'#[\w-]+\s*\{[^}]*\}', '', text, flags=re.DOTALL)  # #id { ... }
        text = re.sub(r'\.[\w-]+\s*\{[^}]*\}', '', text, flags=re.DOTALL)  # .class { ... }
        text = re.sub(r'\.[\w-]+:[\w-]+\s*\{[^}]*\}', '', text, flags=re.DOTALL)  # .class:pseudo { ... }
        text = re.sub(r'\.[\w-]+::[\w-]+\s*\{[^}]*\}', '', text, flags=re.DOTALL)  # .class::pseudo { ... }
        text = re.sub(r'\b(padding|margin|border|background|display|position|height|width|flex|z-index|content|font-size|line-height|transform|webkit)[-\w]*\s*:[^;]+;', '', text, flags=re.IGNORECASE)
        # 更激进的 CSS 清理
        text = re.sub(r'\.tdb?_[\w-]+[^}]*\}', '', text, flags=re.DOTALL)
        text = re.sub(r'@media\s*\(.*?\)\s*\{.*?\}', '', text, flags=re.DOTALL)

        # 2.6 移除更多 JavaScript
        text = re.sub(r'let\s+\w+\s*=.*?;', '', text, flags=re.DOTALL)
        text = re.sub(r'const\s+\w+\s*=.*?;', '', text, flags=re.DOTALL)
        text = re.sub(r'if\s*\([^)]+\)\s*\{[^}]*\}(\s*else\s*\{[^}]*\})?', '', text, flags=re.DOTALL)
        text = re.sub(r'console\.log\([^)]*\);?', '', text)

        # 2.7 移除斜杠包围的标记（如 /ТАСС/, /Reuters/ 等）
        text = re.sub(r'/[A-Za-zА-Яа-яёЁ]+/', '', text)

        # 3. 移除 HTML 标签
        text = re.sub(r'<[^>]+>', ' ', text)

        # 4. 解码 HTML 实体 (如 &amp; -> &, &lt; -> <, &#39; -> ')
        text = html.unescape(text)

        # 5. 移除 URL
        text = re.sub(r'https?://\S+', '', text)

        # 6. 移除特殊字符和控制字符（包括C1控制字符）
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # 7. 规范化空白字符
        text = re.sub(r'[\t\r]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # 8. 移除常见噪音模式
        noise_patterns = [
            r'責任編輯[：:]\s*\S+',
            r'责任编辑[：:]\s*\S+',
            r'Follow us on:.*',
            r'Short link:.*',
            r'相關文章.*',
            r'推薦閱讀.*',
            r'【字號】[\s\S]*?(?:大\s*中\s*小|小\s*中\s*大)',
            r'字體[：:].*',
            r'打印版.*',
            r'转发到.*',
            r'分享到.*',
            r'Related:.*',
            r'Tags:.*',
            r'Subscribe.*newsletter.*',
            r'Advertisement.*',
            r'ADVERTISEMENT.*',
        ]
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # 9. 最终清理空白
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n[ \t]+', '\n', text)
        text = re.sub(r'[ \t]+\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _fix_char_by_char_newlines(self, text: str) -> str:
        """
        修复每个字符被换行符分隔的问题
        例如: "大\n紀\n元" -> "大紀元"
              "B\nT\nS\n \nP\nE\nR\nM" -> "BTS PERM"
        """
        lines = text.split('\n')

        # 检测是否是单字符换行模式
        single_char_lines = sum(1 for line in lines if len(line) <= 1)

        if single_char_lines > len(lines) * 0.5:
            result = []
            buffer = []

            for line in lines:
                if len(line) == 1:
                    char = line
                    if char == ' ':
                        if buffer:
                            result.append(''.join(buffer))
                            buffer = []
                        result.append(' ')
                    else:
                        buffer.append(char)
                elif len(line) == 0:
                    if buffer:
                        result.append(''.join(buffer))
                        buffer = []
                    result.append('\n')
                else:
                    if buffer:
                        result.append(''.join(buffer))
                        buffer = []
                    result.append(line.strip())

            if buffer:
                result.append(''.join(buffer))

            text = ''.join(result)
            text = re.sub(r' +', ' ', text)
            text = re.sub(r'\n +', '\n', text)
            text = re.sub(r' +\n', '\n', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            return text

        return text

    def compute_hash(self, title: str, content: str) -> str:
        """计算内容hash用于去重"""
        text = f"{title}|{content[:200]}"
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def is_garbled(self, text: str) -> bool:
        """
        检测文本是否是乱码
        """
        if not text:
            return False

        replacement_chars = text.count('�')
        garbled_patterns = len(re.findall(r'[�□■◆●▲★☆♦♣♥♠]{2,}', text))
        non_printable = len(re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f]{2,}', text))

        total_issues = replacement_chars + garbled_patterns * 3 + non_printable * 2

        if len(text) > 0 and total_issues / len(text) > 0.1:
            return True
        if replacement_chars > 5:
            return True

        return False

    def is_mostly_code(self, text: str) -> bool:
        """检测文本是否主要是代码（CSS/JS）"""
        if not text or len(text) < 20:
            return False

        code_patterns = [
            r'@media',
            r'\{[^}]*:[^}]*\}',
            r'\.[\w-]+\s*\{',
            r'#[\w-]+\s*\{',
            r'console\.',
            r'function\s*\(',
            r'var\s+\w+\s*=',
            r'let\s+\w+\s*=',
            r'const\s+\w+\s*=',
            r':\s*\d+px',
            r'!important',
        ]

        code_matches = sum(len(re.findall(p, text, re.IGNORECASE)) for p in code_patterns)

        if code_matches > 3:
            return True

        special_chars = len(re.findall(r'[{};:]', text))
        if len(text) > 0 and special_chars / len(text) > 0.1:
            return True

        return False

    def clean_item(self, item: Dict) -> Optional[CleanedItem]:
        """清洗单条数据"""
        url = item.get('articleLink', item.get('url', ''))
        title = item.get('title', '')
        content = item.get('text', item.get('content', ''))
        publish_time = item.get('publishTime', item.get('publish_time', ''))

        if self.is_garbled(title) or self.is_garbled(content):
            self.stats["garbled"] += 1
            return None

        cleaned_title = self.clean_text(title)
        cleaned_content = self.clean_text(content)

        if self.is_garbled(cleaned_title) or self.is_garbled(cleaned_content):
            self.stats["garbled"] += 1
            return None

        if not cleaned_title and not cleaned_content:
            self.stats["empty"] += 1
            return None

        item_id = self.compute_hash(cleaned_title, cleaned_content)

        return CleanedItem(
            id=item_id,
            title=cleaned_title,
            url=url,
            content=cleaned_content,
            publish_time=publish_time,
        )


# =============================================================================
# 便捷函数
# =============================================================================

def truncate_text(text: str, max_length: int = 500) -> str:
    """
    截断文本到指定长度
    - 中文：按字符数
    - 英文：按单词数
    - 同时把换行替换为空格
    """
    if not text:
        return ""

    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()

    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    is_chinese = chinese_chars > len(text) * 0.3

    if is_chinese:
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    else:
        words = text.split()
        if len(words) <= max_length:
            return text
        return " ".join(words[:max_length]) + "..."


def clean_raw_data(input_path: str, output_path: str = None, max_text_length: int = 500):
    """
    清洗原始数据的便捷函数

    Args:
        input_path: 输入文件路径（JSON 或 JSONL）
        output_path: 输出文件路径（默认自动生成）
        max_text_length: 正文最大长度

    Returns:
        统计信息字典
    """
    input_path = Path(input_path)

    # 自动生成输出路径
    if output_path is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "cleaned"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_cleaned.jsonl"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" 数据清洗")
    print("=" * 60)
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print("=" * 60)

    # 加载数据
    print("\n[1/3] 加载数据...")
    with open(input_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    print(f"  共 {len(data)} 条原始数据")

    # 清洗数据
    print("\n[2/3] 清洗数据...")
    cleaner = DataCleaner()

    seen_hashes = set()
    cleaned_items = []
    stats = {
        "total": len(data),
        "empty": 0,
        "garbled": 0,
        "duplicate": 0,
        "cleaned": 0,
    }

    for i, item in enumerate(data):
        if (i + 1) % 1000 == 0:
            print(f"  处理进度: {i + 1}/{len(data)}")

        article_link = item.get('articleLink', item.get('url', ''))
        title = item.get('title', '')
        text = item.get('text', item.get('content', ''))
        publish_time = item.get('publishTime', item.get('publish_time', ''))

        # 检测乱码
        if cleaner.is_garbled(title) or cleaner.is_garbled(text):
            stats["garbled"] += 1
            continue

        # 清洗
        cleaned_title = cleaner.clean_text(title)
        cleaned_text = cleaner.clean_text(text)

        # 清洗后再检测乱码
        if cleaner.is_garbled(cleaned_title) or cleaner.is_garbled(cleaned_text):
            stats["garbled"] += 1
            continue

        # 检查是否为空
        if not cleaned_title and not cleaned_text:
            stats["empty"] += 1
            continue

        # 去重
        content_hash = cleaner.compute_hash(cleaned_title, cleaned_text)
        if content_hash in seen_hashes:
            stats["duplicate"] += 1
            continue
        seen_hashes.add(content_hash)

        # 截断正文
        truncated_text = truncate_text(cleaned_text, max_text_length)

        cleaned_items.append({
            "articleLink": article_link,
            "title": cleaned_title,
            "text": truncated_text,
            "publishTime": publish_time,
        })
        stats["cleaned"] += 1

    # 保存结果
    print("\n[3/3] 保存结果...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in cleaned_items:
            f.write(json.dumps(item, ensure_ascii=False, indent=2) + '\n\n')

    # 打印统计
    print("\n" + "=" * 60)
    print(" 清洗统计")
    print("=" * 60)
    print(f"原始数据:   {stats['total']:,} 条")
    print(f"清洗后:     {stats['cleaned']:,} 条")
    print(f"空数据:     {stats['empty']:,} 条")
    print(f"乱码数据:   {stats['garbled']:,} 条")
    print(f"重复数据:   {stats['duplicate']:,} 条")
    print(f"\n输出文件: {output_path}")
    print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return stats


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="数据清洗工具")
    parser.add_argument("input", nargs="?", help="输入文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径")
    parser.add_argument("--max-length", type=int, default=500, help="正文最大长度")

    args = parser.parse_args()

    # 默认处理的文件
    if args.input is None:
        default_input = Path(__file__).parent.parent.parent / "data" / "raw" / "part-00000-15e62120-d46b-4512-a366-c91dfab91939.json"
        if default_input.exists():
            args.input = str(default_input)
        else:
            print("请指定输入文件路径")
            return

    clean_raw_data(args.input, args.output, args.max_length)


if __name__ == "__main__":
    main()
