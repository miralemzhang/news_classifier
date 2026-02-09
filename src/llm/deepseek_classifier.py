"""
基于 DeepSeek-R1 的网页分类器
使用生成式推理进行分类
"""

import torch
import time
import re
import json
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class DeepSeekClassifyResult:
    """分类结果"""
    label: str
    confidence: float
    content_type: str
    thinking: str  # 模型的思考过程
    raw_output: str
    latency_ms: float


# 默认模型路径
DEFAULT_MODEL_PATH = "/home/zzh/webpage-classification/models/DeepSeek-R1-Distill-Qwen-7B"

# 默认 prompt 路径
DEFAULT_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "classification_prompt.txt"


class DeepSeekClassifier:
    """基于 DeepSeek-R1 的分类器"""

    def __init__(
        self,
        model_path: str = None,
        prompt_path: str = None,
        device: str = None,
        torch_dtype=None,
    ):
        """
        初始化分类器

        Args:
            model_path: 模型路径
            prompt_path: prompt 文件路径
            device: 设备 (cuda/cpu)
            torch_dtype: 数据类型
        """
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # 加载 prompt 模板
        prompt_path = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
            print(f"已加载 prompt: {prompt_path}")
        else:
            raise FileNotFoundError(f"Prompt 文件不存在: {prompt_path}")

        print(f"正在加载 DeepSeek 模型: {model_path}")
        print(f"设备: {device}, 数据类型: {torch_dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        self.device = device

        # 有效类别
        self.categories = ["时政", "经济", "军事", "社会", "科技", "体育", "娱乐", "其他"]

        # 日志记录
        self.log_entries = []

        print("DeepSeek 模型加载完成!")

    def _parse_output(self, output: str) -> tuple:
        """
        解析模型输出，提取 thinking 和分类结果

        Returns:
            (content_type, label, thinking)
        """
        # 提取 thinking 部分 (DeepSeek-R1 格式: <think>...</think>)
        thinking = ""
        think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()

        # 匹配格式: content_type: xxx | label: xxx
        pattern = r'content_type:\s*(\w+)\s*\|\s*label:\s*(\S+)'
        match = re.search(pattern, output)

        if match:
            content_type = match.group(1)
            label = match.group(2)
            # 清理标签
            label = label.strip('*').strip()
            if label not in self.categories:
                label = "其他"
            return content_type, label, thinking

        # 备用解析：直接查找类别关键词
        for cat in self.categories:
            if cat in output:
                return "unknown", cat, thinking

        return "unknown", "其他", thinking

    def classify(
        self,
        title: str,
        url: str = "",
        content: str = "",
        item_id: int = None,
    ) -> DeepSeekClassifyResult:
        """
        对输入文本进行分类

        Args:
            title: 标题
            url: URL
            content: 正文内容
            item_id: 数据项 ID（用于日志）

        Returns:
            DeepSeekClassifyResult: 分类结果
        """
        start_time = time.time()

        # 使用外部 prompt 模板
        prompt = self.prompt_template.format(
            title=title or "无",
            url=url or "无",
            content=content[:500] if content else "无"
        )

        # 构建消息
        messages = [{"role": "user", "content": prompt}]

        # 生成
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=512,  # 增加以容纳 thinking
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码输出
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # 解析结果
        content_type, label, thinking = self._parse_output(response)

        latency_ms = (time.time() - start_time) * 1000

        result = DeepSeekClassifyResult(
            label=label,
            confidence=0.8 if content_type != "unknown" else 0.5,
            content_type=content_type,
            thinking=thinking,
            raw_output=response.strip(),
            latency_ms=latency_ms
        )

        # 记录日志
        self.log_entries.append({
            "id": item_id,
            "title": title,
            "url": url,
            "label": label,
            "content_type": content_type,
            "confidence": result.confidence,
            "thinking": thinking,
            "raw_output": response.strip(),
            "latency_ms": latency_ms
        })

        return result

    def batch_classify(
        self,
        items: List[Dict],
        batch_size: int = 1,  # DeepSeek 生成式模型一般逐条处理
        start_id: int = 0,  # 起始 ID
        verbose: bool = True,  # 是否输出详细日志
    ) -> List[DeepSeekClassifyResult]:
        """
        批量分类

        Args:
            items: 输入列表
            batch_size: 批大小（生成式模型通常为1）
            start_id: 起始 ID
            verbose: 是否输出详细日志（thinking 过程）

        Returns:
            List[DeepSeekClassifyResult]: 分类结果列表
        """
        results = []
        total = len(items)

        if verbose:
            print(f"\n  {'='*60}")
            print(f"  DeepSeek 实时推理日志 (共 {total} 条)")
            print(f"  {'='*60}")

        for i, item in enumerate(items):
            title = item.get("title", "")
            url = item.get("articleLink", item.get("url", ""))
            content = item.get("text", item.get("content", ""))
            item_id = item.get("id", start_id + i)

            result = self.classify(
                title=title,
                url=url,
                content=content,
                item_id=item_id
            )
            results.append(result)

            # 实时输出推理过程
            if verbose:
                print(f"\n  [{item_id:4d}] {title[:40]}")
                print(f"  结果: {result.label} | 类型: {result.content_type} | 置信度: {result.confidence:.2f}")
                if result.thinking:
                    # 截取 thinking 的关键部分（前200字符）
                    thinking_preview = result.thinking[:200].replace('\n', ' ')
                    if len(result.thinking) > 200:
                        thinking_preview += "..."
                    print(f"  思考: {thinking_preview}")
                print(f"  {'-'*58}")

            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"  进度: {i + 1}/{total}")

        if verbose:
            print(f"  {'='*60}\n")

        return results

    def save_log(self, output_path: Path) -> Path:
        """
        保存推理日志

        Args:
            output_path: 结果输出路径

        Returns:
            日志文件路径
        """
        # 生成日志文件路径
        log_path = output_path.with_name(
            output_path.stem + "_deepseek_log.json"
        )

        # 构建日志内容
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "total_items": len(self.log_entries),
            "prompt_file": str(DEFAULT_PROMPT_PATH),
            "entries": self.log_entries
        }

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        print(f"  推理日志已保存: {log_path}")
        return log_path

    def clear_log(self):
        """清空日志"""
        self.log_entries = []


def create_deepseek_classifier(
    model_path: str = None,
    device: str = None
) -> DeepSeekClassifier:
    """
    创建 DeepSeek 分类器的工厂函数

    Args:
        model_path: 模型路径
        device: 设备

    Returns:
        DeepSeekClassifier: 分类器实例
    """
    return DeepSeekClassifier(model_path=model_path, device=device)


if __name__ == "__main__":
    # 测试代码
    classifier = create_deepseek_classifier()

    result = classifier.classify(
        title="中国人民银行宣布下调贷款利率",
        content="央行今日宣布，将一年期贷款市场报价利率下调10个基点...",
        item_id=0
    )

    print(f"分类结果: {result.label}")
    print(f"内容类型: {result.content_type}")
    print(f"置信度: {result.confidence:.2%}")
    print(f"耗时: {result.latency_ms:.2f}ms")
    print(f"\n思考过程:\n{result.thinking}")
    print(f"\n原始输出:\n{result.raw_output}")
