"""
HTML 内容抽取模块

从 HTML 中提取分类所需的关键信息：
- title: 页面标题
- meta_description: meta 描述
- meta_keywords: meta 关键词
- h1: 主标题
- content: 正文内容
- url_path: URL 路径（强特征）
- breadcrumb: 面包屑导航
"""

import re
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from urllib.parse import urlparse

# pip install beautifulsoup4 lxml trafilatura
from bs4 import BeautifulSoup
import trafilatura


@dataclass
class ExtractedContent:
    """抽取结果数据类"""
    url: str
    title: str
    meta_description: str
    meta_keywords: str
    h1: str
    h2_list: list[str]
    content: str
    url_path: str
    breadcrumb: str
    
    def to_dict(self):
        return asdict(self)
    
    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)


class HTMLExtractor:
    """HTML 内容抽取器"""
    
    def __init__(self, max_content_length: int = 2000):
        """
        Args:
            max_content_length: 正文最大长度（字符数）
        """
        self.max_content_length = max_content_length
    
    def extract(self, html: str, url: str = "") -> ExtractedContent:
        """
        从 HTML 中抽取结构化内容
        
        Args:
            html: HTML 源码
            url: 页面 URL
        
        Returns:
            ExtractedContent 对象
        """
        soup = BeautifulSoup(html, 'lxml')
        
        # 1. 提取 title
        title = self._extract_title(soup)
        
        # 2. 提取 meta 标签
        meta_desc = self._extract_meta(soup, 'description')
        meta_keywords = self._extract_meta(soup, 'keywords')
        
        # 3. 提取 h1, h2
        h1 = self._extract_h1(soup)
        h2_list = self._extract_h2_list(soup)
        
        # 4. 提取正文（使用 trafilatura）
        content = self._extract_content(html)
        
        # 5. 提取 URL 路径
        url_path = self._extract_url_path(url)
        
        # 6. 提取面包屑
        breadcrumb = self._extract_breadcrumb(soup)
        
        return ExtractedContent(
            url=url,
            title=title,
            meta_description=meta_desc,
            meta_keywords=meta_keywords,
            h1=h1,
            h2_list=h2_list,
            content=content,
            url_path=url_path,
            breadcrumb=breadcrumb
        )
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """提取页面标题"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        return ""
    
    def _extract_meta(self, soup: BeautifulSoup, name: str) -> str:
        """提取 meta 标签内容"""
        meta_tag = soup.find('meta', attrs={'name': name})
        if meta_tag:
            return meta_tag.get('content', '')
        return ""
    
    def _extract_h1(self, soup: BeautifulSoup) -> str:
        """提取第一个 h1"""
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text(strip=True)
        return ""
    
    def _extract_h2_list(self, soup: BeautifulSoup, max_count: int = 5) -> list[str]:
        """提取 h2 列表"""
        h2_tags = soup.find_all('h2', limit=max_count)
        return [h2.get_text(strip=True) for h2 in h2_tags]
    
    def _extract_content(self, html: str) -> str:
        """使用 trafilatura 提取正文"""
        content = trafilatura.extract(html)
        if content:
            # 截断到最大长度
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "..."
            return content
        return ""
    
    def _extract_url_path(self, url: str) -> str:
        """提取 URL 路径（分类强特征）"""
        if not url:
            return ""
        try:
            parsed = urlparse(url)
            return parsed.path
        except:
            return ""
    
    def _extract_breadcrumb(self, soup: BeautifulSoup) -> str:
        """提取面包屑导航"""
        # 常见的面包屑 class
        breadcrumb_selectors = [
            '.breadcrumb',
            '.crumb',
            '.nav-path',
            '[class*="breadcrumb"]',
            '[class*="crumb"]'
        ]
        
        for selector in breadcrumb_selectors:
            try:
                elem = soup.select_one(selector)
                if elem:
                    text = elem.get_text(separator=' > ', strip=True)
                    # 清理多余空格
                    text = re.sub(r'\s+', ' ', text)
                    return text
            except:
                continue
        
        return ""


def batch_extract(input_dir: str, output_file: str):
    """
    批量抽取 HTML 文件
    
    Args:
        input_dir: 输入目录（包含 HTML 文件）
        output_file: 输出文件（JSONL 格式）
    """
    extractor = HTMLExtractor()
    input_path = Path(input_dir)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for html_file in input_path.glob('*.html'):
            try:
                html_content = html_file.read_text(encoding='utf-8')
                # 从文件名推断 URL（实际项目需要从元数据获取）
                url = f"https://example.com/{html_file.stem}"
                
                result = extractor.extract(html_content, url)
                f.write(result.to_json() + '\n')
                
            except Exception as e:
                print(f"Error processing {html_file}: {e}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python extractor.py <input_dir> <output_file>")
        print("Example: python extractor.py data/raw data/extracted/extracted.jsonl")
        sys.exit(1)
    
    batch_extract(sys.argv[1], sys.argv[2])
