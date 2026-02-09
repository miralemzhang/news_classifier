"""
基于 LoRA 微调 DeepSeek 的网页分类器
使用 PEFT 库进行参数高效微调
"""

import torch
import time
import re
import json
import sys
import argparse
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training,
)
from datasets import Dataset
import numpy as np


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


@dataclass
class LoRAClassifyResult:
    """分类结果"""
    label: str
    confidence: float
    raw_output: str
    latency_ms: float


# 默认模型路径
DEFAULT_MODEL_PATH = "/home/zzh/webpage-classification/models/DeepSeek-R1-Distill-Qwen-7B"

# 类别定义
LABELS = ["时政", "经济", "军事", "社会", "科技", "体育", "娱乐", "其他"]

# LoRA 适配器默认保存路径
DEFAULT_LORA_PATH = "/home/zzh/webpage-classification/models/deepseek_lora_classifier"


class DeepSeekLoRAClassifier:
    """基于 LoRA 微调的 DeepSeek 分类器"""

    def __init__(
        self,
        model_path: str = None,
        lora_path: str = None,
        device: str = None,
        torch_dtype=None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        初始化分类器

        Args:
            model_path: 基础模型路径
            lora_path: LoRA 适配器路径（如果已训练）
            device: 设备 (cuda/cpu)
            torch_dtype: 数据类型
            load_in_8bit: 是否使用 8-bit 量化
            load_in_4bit: 是否使用 4-bit 量化
        """
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.categories = LABELS

        print(f"正在加载 DeepSeek 模型: {model_path}")
        print(f"设备: {device}, 数据类型: {torch_dtype}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",  # 生成时需要左填充
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }

        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        elif load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs
        )

        # 如果提供了 LoRA 路径，加载适配器
        if lora_path and Path(lora_path).exists():
            print(f"加载 LoRA 适配器: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.is_trained = True
        else:
            self.is_trained = False

        self.model.eval()
        print("模型加载完成!")

    def _build_prompt(self, title: str, content: str = "") -> str:
        """构建分类 prompt"""
        prompt = f"""将以下新闻分类到最合适的类别。

可选类别：时政、经济、军事、社会、科技、体育、娱乐、其他

类别说明：
- 时政：国家领导人活动、政府政策、外交访问、国际关系
- 经济：股票市场、企业财报、银行利率、贸易进出口
- 军事：军队演习、武器装备、战争冲突、军事行动
- 社会：民生新闻、社会热点、教育医疗、犯罪案件
- 科技：人工智能、互联网、手机数码、科学研究
- 体育：足球篮球、体育赛事、奥运会、世界杯
- 娱乐：明星八卦、影视综艺、音乐演唱会
- 其他：不属于以上任何类别

新闻标题：{title}
"""
        if content:
            # 限制内容长度
            content = content[:300]
            prompt += f"新闻内容：{content}\n"

        prompt += "\n请只输出一个类别名称："
        return prompt

    def _parse_label(self, output: str) -> str:
        """从模型输出中解析类别"""
        output = output.strip()

        # 直接匹配类别
        for label in self.categories:
            if label in output:
                return label

        # 如果没有匹配到，返回"其他"
        return "其他"

    def classify(
        self,
        title: str,
        url: str = "",
        content: str = "",
    ) -> LoRAClassifyResult:
        """
        对输入文本进行分类

        Args:
            title: 标题
            url: URL（未使用）
            content: 正文内容

        Returns:
            LoRAClassifyResult: 分类结果
        """
        start_time = time.time()

        prompt = self._build_prompt(title, content)

        # 构建消息
        messages = [{"role": "user", "content": prompt}]

        # 生成
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=16,  # 只需要输出类别名
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码输出
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        label = self._parse_label(response)
        latency_ms = (time.time() - start_time) * 1000

        # 简单的置信度估计
        confidence = 0.9 if label != "其他" else 0.5

        return LoRAClassifyResult(
            label=label,
            confidence=confidence,
            raw_output=response,
            latency_ms=latency_ms
        )

    def batch_classify(
        self,
        items: List[Dict],
        verbose: bool = True,
    ) -> List[LoRAClassifyResult]:
        """
        批量分类

        Args:
            items: 输入列表
            verbose: 是否输出进度

        Returns:
            List[LoRAClassifyResult]: 分类结果列表
        """
        results = []
        total = len(items)

        for i, item in enumerate(items):
            title = item.get("title", "")
            url = item.get("articleLink", item.get("url", ""))
            content = item.get("text", item.get("content", ""))

            result = self.classify(title=title, url=url, content=content)
            results.append(result)

            if verbose and ((i + 1) % 10 == 0 or (i + 1) == total):
                print(f"  进度: {i + 1}/{total}")

        return results

    def prepare_training_data(
        self,
        items: List[Dict],
        max_length: int = 512,
    ) -> Dataset:
        """
        准备训练数据

        Args:
            items: 带标签的数据列表
            max_length: 最大序列长度

        Returns:
            Dataset: HuggingFace Dataset
        """

        def tokenize_function(examples):
            """Tokenize 并创建标签"""
            model_inputs = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }

            for i in range(len(examples["title"])):
                title = examples["title"][i]
                content = examples.get("text", examples.get("content", [""] * len(examples["title"])))[i] or ""
                label = examples["label"][i]

                # 构建输入
                prompt = self._build_prompt(title, content)
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": label}
                ]

                # Tokenize 完整对话
                full_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )

                # 只 tokenize 用户输入部分（用于计算 labels mask）
                user_messages = [{"role": "user", "content": prompt}]
                user_text = self.tokenizer.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Tokenize
                full_tokenized = self.tokenizer(
                    full_text,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )

                user_tokenized = self.tokenizer(
                    user_text,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )

                input_ids = full_tokenized["input_ids"]
                attention_mask = full_tokenized["attention_mask"]

                # 创建 labels：用户输入部分设为 -100（不计算 loss）
                labels = input_ids.copy()
                user_len = len(user_tokenized["input_ids"])
                labels[:user_len] = [-100] * user_len

                model_inputs["input_ids"].append(input_ids)
                model_inputs["attention_mask"].append(attention_mask)
                model_inputs["labels"].append(labels)

            return model_inputs

        # 转换为 Dataset
        data_dict = {
            "title": [item.get("title", "") for item in items],
            "text": [item.get("text", item.get("content", "")) for item in items],
            "label": [item.get("label", "其他") for item in items],
        }
        dataset = Dataset.from_dict(data_dict)

        # Tokenize
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )

        return tokenized_dataset

    def train(
        self,
        train_data: List[Dict],
        eval_data: List[Dict] = None,
        output_dir: str = None,
        # LoRA 参数
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: List[str] = None,
        # 训练参数
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.1,
        max_length: int = 512,
        save_steps: int = 100,
        logging_steps: int = 10,
    ):
        """
        使用 LoRA 微调模型

        Args:
            train_data: 训练数据
            eval_data: 评估数据
            output_dir: 输出目录
            lora_r: LoRA 秩
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            target_modules: 目标模块
            num_epochs: 训练轮数
            batch_size: 批大小
            gradient_accumulation_steps: 梯度累积步数
            learning_rate: 学习率
            warmup_ratio: warmup 比例
            max_length: 最大序列长度
            save_steps: 保存步数
            logging_steps: 日志步数
        """
        if output_dir is None:
            output_dir = DEFAULT_LORA_PATH

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"开始 LoRA 微调")
        print(f"{'='*60}")
        print(f"训练数据: {len(train_data)} 条")
        print(f"评估数据: {len(eval_data) if eval_data else 0} 条")
        print(f"输出目录: {output_dir}")
        print(f"LoRA 参数: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        print(f"训练参数: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
        print(f"{'='*60}\n")

        # 准备训练数据
        print("准备训练数据...")
        train_dataset = self.prepare_training_data(train_data, max_length)
        eval_dataset = None
        if eval_data:
            eval_dataset = self.prepare_training_data(eval_data, max_length)

        # 配置 LoRA
        if target_modules is None:
            # DeepSeek/Qwen 的典型目标模块
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # 准备模型
        self.model.train()
        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # 应用 LoRA
        print("应用 LoRA 适配器...")
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # 训练参数
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=2,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            bf16=self.torch_dtype == torch.bfloat16,
            fp16=self.torch_dtype == torch.float16,
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )

        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            pad_to_multiple_of=8,
            padding=True,
        )

        # 创建 Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # 开始训练
        print("\n开始训练...")
        train_result = trainer.train()

        # 保存模型
        print(f"\n保存 LoRA 适配器到: {output_dir}")
        self.model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))

        # 保存训练信息
        train_info = {
            "timestamp": datetime.now().isoformat(),
            "train_samples": len(train_data),
            "eval_samples": len(eval_data) if eval_data else 0,
            "lora_config": {
                "r": lora_r,
                "alpha": lora_alpha,
                "dropout": lora_dropout,
                "target_modules": target_modules,
            },
            "training_args": {
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
            "train_result": {
                "loss": train_result.training_loss,
                "global_step": train_result.global_step,
            }
        }

        with open(output_dir / "train_info.json", "w", encoding="utf-8") as f:
            json.dump(train_info, f, ensure_ascii=False, indent=2)

        print(f"\n训练完成! Loss: {train_result.training_loss:.4f}")

        self.model.eval()
        self.is_trained = True

        return train_result


def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 尝试多行 JSON 格式
    if '}\n{' in content or '}\r\n{' in content:
        # 标准 JSONL
        for line in content.strip().split('\n'):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    else:
        # 可能是多行格式的 JSON 数组或对象
        try:
            # 尝试解析为 JSON 数组
            parsed = json.loads(content)
            if isinstance(parsed, list):
                data = parsed
            else:
                data = [parsed]
        except json.JSONDecodeError:
            # 尝试逐个解析 JSON 对象
            import re
            pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    data.append(json.loads(match))
                except json.JSONDecodeError:
                    pass

    return data


def evaluate(
    classifier: DeepSeekLoRAClassifier,
    test_data: List[Dict],
    verbose: bool = True,
) -> Dict:
    """
    评估分类器

    Args:
        classifier: 分类器
        test_data: 测试数据
        verbose: 是否输出详细信息

    Returns:
        评估结果字典
    """
    if verbose:
        print(f"\n开始评估 ({len(test_data)} 条数据)...")

    results = classifier.batch_classify(test_data, verbose=verbose)

    # 统计结果
    correct = 0
    total = len(test_data)
    predictions = []
    labels = []
    errors = []

    for i, (item, result) in enumerate(zip(test_data, results)):
        true_label = item.get("label", "其他")
        pred_label = result.label

        predictions.append(pred_label)
        labels.append(true_label)

        if pred_label == true_label:
            correct += 1
        else:
            errors.append({
                "id": item.get("id", i),
                "title": item.get("title", ""),
                "true_label": true_label,
                "pred_label": pred_label,
            })

    accuracy = correct / total if total > 0 else 0

    # 按类别统计
    from collections import defaultdict
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for true, pred in zip(labels, predictions):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1

    class_accuracy = {
        label: class_correct[label] / class_total[label]
        if class_total[label] > 0 else 0
        for label in LABELS
    }

    result = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "class_accuracy": class_accuracy,
        "errors": errors[:20],  # 只保留前20个错误
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"评估结果")
        print(f"{'='*60}")
        print(f"总体准确率: {accuracy:.2%} ({correct}/{total})")
        print(f"\n各类别准确率:")
        for label in LABELS:
            if class_total[label] > 0:
                acc = class_accuracy[label]
                print(f"  {label}: {acc:.2%} ({class_correct[label]}/{class_total[label]})")
        print(f"{'='*60}\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="DeepSeek LoRA 分类器")
    parser.add_argument("--data", type=str, default="/home/zzh/webpage-classification/data/labeled/labeled.jsonl",
                       help="数据文件路径")
    parser.add_argument("--train", type=int, default=500, help="训练数据量")
    parser.add_argument("--test", type=int, default=100, help="测试数据量")
    parser.add_argument("--load", type=str, default=None, help="加载已训练的 LoRA 适配器")
    parser.add_argument("--output", type=str, default=None, help="输出目录")

    # LoRA 参数
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA 秩")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="批大小")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="梯度累积步数")

    # 量化选项
    parser.add_argument("--load-in-8bit", action="store_true", help="使用 8-bit 量化")
    parser.add_argument("--load-in-4bit", action="store_true", help="使用 4-bit 量化")

    args = parser.parse_args()

    # 自动保存日志
    result_dir = Path("/home/zzh/webpage-classification/data/results")
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = result_dir / f"deepseek_lora_train{args.train}_test{args.test}_{timestamp}.log"
    logger = TeeLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    print(f"Log 保存至: {log_path}")

    print("=" * 70)
    print(" DeepSeek LoRA 分类器 - 训练与评估")
    print("=" * 70)
    print(f"数据文件: {args.data}")
    print(f"训练样本: {args.train} 条")
    print(f"测试样本: {args.test} 条")
    print(f"LoRA 参数: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"训练参数: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    print("=" * 70)

    # 加载数据
    print(f"\n加载数据: {args.data}")
    all_data = load_jsonl(args.data)
    print(f"总数据量: {len(all_data)}")

    # 过滤有标签的数据
    labeled_data = [item for item in all_data if item.get("label")]
    print(f"有标签数据: {len(labeled_data)}")

    # 划分数据
    train_data = labeled_data[:args.train]
    test_data = labeled_data[args.train:args.train + args.test]

    print(f"训练数据: {len(train_data)}")
    print(f"测试数据: {len(test_data)}")

    # 创建分类器
    classifier = DeepSeekLoRAClassifier(
        lora_path=args.load,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    # 如果没有加载已训练的模型，进行训练
    lora_output_dir = None
    if not args.load:
        lora_output_dir = args.output
        if lora_output_dir is None:
            lora_output_dir = f"/home/zzh/webpage-classification/models/deepseek_lora_train{args.train}_{timestamp}"

        classifier.train(
            train_data=train_data,
            eval_data=test_data[:20] if len(test_data) >= 20 else None,
            output_dir=lora_output_dir,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.lr,
        )
        print(f"\nLoRA 模型已保存至: {lora_output_dir}")

    # 评估
    if test_data:
        eval_result = evaluate(classifier, test_data)

        # 构建完整结果数据
        result_data = {
            'timestamp': timestamp,
            'config': {
                'method': 'DeepSeek LoRA Fine-tuning',
                'train_count': args.train,
                'test_count': args.test,
                'lora_r': args.lora_r,
                'lora_alpha': args.lora_alpha,
                'lora_dropout': args.lora_dropout,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
            },
            'overall': {
                'accuracy': eval_result['accuracy'],
                'correct': eval_result['correct'],
                'total': eval_result['total'],
            },
            'per_class': {
                label: {
                    'accuracy': eval_result['class_accuracy'].get(label, 0),
                }
                for label in LABELS
            },
            'errors': eval_result['errors'],
        }

        # 保存到 data/results 目录
        result_filename = f"deepseek_lora_train{args.train}_test{args.test}_{timestamp}.json"
        result_path = result_dir / result_filename
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存至: {result_path}")

        # 也保存到模型目录
        if lora_output_dir:
            model_result_path = Path(lora_output_dir) / "eval_result.json"
            with open(model_result_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    logger.close()
    sys.stdout = logger.terminal


if __name__ == "__main__":
    main()
