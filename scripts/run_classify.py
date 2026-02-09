#!/usr/bin/env python3
"""
网页分类脚本

分类流程：
1. 规则分类：尝试所有语言的关键词库
2. 规则无法匹配 → 使用指定模型分类
3. 模型置信度低 → 仍使用模型结果，标记为低置信度

支持的模型：
- qwen (默认): Qwen3-VL-Embedding，使用嵌入相似度分类
- deepseek: DeepSeek-R1，使用生成式推理分类

使用:
    python run_classify.py                              # 使用默认输入和模型
    python run_classify.py -i data/cleaned/xxx.jsonl   # 指定输入文件
    python run_classify.py --model deepseek            # 使用 DeepSeek 模型
    python run_classify.py --model-path /path/to/model # 指定模型路径
    python run_classify.py --limit 100                  # 测试：只处理前 100 条
"""

import sys
import json
import time
import argparse
from pathlib import Path
from collections import Counter

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.rules.classifier import RuleClassifier, SUPPORTED_LANGUAGES


# 默认输入文件
DEFAULT_INPUT = PROJECT_ROOT / "data/cleaned/top400.jsonl"


def get_next_output_path(base_name: str = "top200_result") -> Path:
    """获取下一个可用的输出路径（自动编号）"""
    output_dir = PROJECT_ROOT / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    num = 1
    while True:
        output_path = output_dir / f"{base_name}_{num:02d}.json"
        if not output_path.exists():
            return output_path
        num += 1


def load_data(path: Path) -> list:
    """加载数据（支持 JSON、JSONL 和多行 JSON 格式）"""
    import re

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    first_char = content.strip()[0] if content.strip() else ''

    if first_char == '[':
        # JSON 数组格式
        return json.loads(content)
    elif first_char == '{':
        # 尝试解析多行 JSON 对象（每个对象可能跨多行）
        objects = []
        depth = 0
        start = -1
        for i, char in enumerate(content):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    obj_str = content[start:i+1]
                    # 修复 trailing comma
                    obj_str = re.sub(r',(\s*})', r'\1', obj_str)
                    try:
                        obj = json.loads(obj_str)
                        objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = -1
        return objects
    else:
        # 标准 JSONL 格式（每行一个 JSON）
        return [json.loads(line) for line in content.split('\n') if line.strip()]


def classify(
    input_path: str = None,
    output_path: str = None,
    limit: int = None,
    model_type: str = "qwen",
    model_path: str = None,
    start_id: int = None,
    end_id: int = None,
):
    """
    对已清洗的数据进行分类

    流程：
    1. 支持的语言 → 规则分类
    2. 其他语言 / 规则无法匹配 → 模型分类

    Args:
        input_path: 输入数据路径（已清洗的 JSONL）
        output_path: 输出结果路径
        limit: 限制处理条数（用于测试）
        model_type: 模型类型 ("qwen" 或 "deepseek")
        model_path: 模型路径（可选，覆盖默认路径）
        start_id: 起始 id（包含）
        end_id: 结束 id（包含）
    """
    input_path = Path(input_path) if input_path else DEFAULT_INPUT

    # 自动生成输出路径（带编号）
    if output_path is None:
        # 从输入文件名生成基础名，或根据 id 范围生成
        if start_id is not None and end_id is not None:
            base_name = f"{start_id}to{end_id}_result"
        else:
            base_name = input_path.stem.replace('_cleaned', '') + '_result'
        output_path = get_next_output_path(base_name)
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # 模型名称映射
    model_names = {
        "qwen": "Qwen3-VL-Embedding",
        "deepseek": "DeepSeek-R1"
    }

    # 打印配置
    print("=" * 70)
    print(" 网页分类 (多语言规则 + 模型兜底)")
    print("=" * 70)
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"支持的语言: {', '.join(sorted(SUPPORTED_LANGUAGES))}")
    print(f"兜底模型: {model_names.get(model_type, model_type)}")
    if model_path:
        print(f"模型路径: {model_path}")
    if start_id is not None or end_id is not None:
        print(f"ID 范围: {start_id if start_id is not None else '起始'} - {end_id if end_id is not None else '结束'}")
    if limit:
        print(f"限制条数: {limit}")
    print("=" * 70)

    # 加载数据
    print("\n[1/4] 加载数据...")
    data = load_data(input_path)

    # 根据 id 范围过滤数据
    if start_id is not None or end_id is not None:
        original_count = len(data)
        data = [
            item for item in data
            if (start_id is None or item.get('id', -1) >= start_id) and
               (end_id is None or item.get('id', -1) <= end_id)
        ]
        print(f"  原始数据: {original_count} 条，过滤后: {len(data)} 条 (id {start_id}-{end_id})")

    if limit:
        data = data[:limit]
    print(f"  共 {len(data)} 条数据")

    # 规则分类
    print("\n[2/4] 规则分类...")
    rule_classifier = RuleClassifier()

    results = []
    rule_matched = 0
    needs_model = []  # 需要模型处理的数据

    start_time = time.time()

    for i, item in enumerate(data):
        if (i + 1) % 500 == 0:
            print(f"  进度: {i + 1}/{len(data)}")

        item_id = item.get('id', i)  # 使用原始 id，如果没有则用索引
        title = item.get('title', '')
        url = item.get('articleLink', item.get('url', ''))
        content = item.get('text', item.get('content', ''))

        # 尝试规则分类（自动检测语言并尝试所有语言关键词）
        rule_result = rule_classifier.classify(
            title=title,
            content=content,
            url_path=url
        )

        # 规则匹配成功且置信度足够高（>=0.80 才直接使用，与 classifier.py 保持一致）
        if rule_result is not None and rule_result.confidence >= 0.80:
            results.append({
                'id': item_id,
                'title': title,
                'url': url,
                'category': rule_result.label,
                'confidence': rule_result.confidence,
                'method': 'rule',
                'matched_keywords': rule_result.matched_keywords[:5],
            })
            rule_matched += 1
        else:
            # 规则置信度不够，交给模型
            needs_model.append((item_id, item, rule_result))

    rule_time = time.time() - start_time
    if len(data) > 0:
        print(f"  规则匹配: {rule_matched}/{len(data)} ({rule_matched/len(data)*100:.1f}%)")
    else:
        print("  没有数据需要处理")
        return
    print(f"  需要模型: {len(needs_model)} 条")
    print(f"  耗时: {rule_time:.1f}s")

    # 模型分类
    print(f"\n[3/4] 模型分类 ({model_names.get(model_type, model_type)})...")

    if needs_model:
        try:
            # 根据模型类型加载不同的分类器
            if model_type == "deepseek":
                from src.llm.deepseek_classifier import create_deepseek_classifier
                print("  加载 DeepSeek-R1 模型...")
                model_classifier = create_deepseek_classifier(model_path=model_path)
            else:
                from src.llm.embedding_classifier import create_classifier
                print("  加载 Qwen3-VL-Embedding 模型...")
                model_classifier = create_classifier(model_path=model_path)

            model_start = time.time()
            model_count = 0

            # 批量处理以提高效率
            batch_items = [item for _, item, _ in needs_model]
            print(f"  批量处理 {len(batch_items)} 条...")

            model_results = model_classifier.batch_classify(batch_items)

            for idx, ((i, item, rule_result), model_result) in enumerate(zip(needs_model, model_results)):
                title = item.get('title', '')
                url = item.get('articleLink', item.get('url', ''))

                # 如果规则有候选结果，参考规则的候选
                final_label = model_result.label
                final_confidence = model_result.confidence

                # 如果规则有低置信度结果，且与模型一致，提高置信度
                if rule_result and rule_result.label == model_result.label:
                    final_confidence = min(0.95, final_confidence + 0.1)

                # 构建结果（根据模型类型）
                result_item = {
                    'id': i,
                    'title': title,
                    'url': url,
                    'category': final_label,
                    'confidence': final_confidence,
                    'method': 'model',
                }

                # DeepSeek 有 content_type，Qwen 有 scores
                if hasattr(model_result, 'content_type'):
                    result_item['content_type'] = model_result.content_type
                if hasattr(model_result, 'scores'):
                    result_item['scores'] = model_result.scores

                results.append(result_item)
                model_count += 1

            model_time = time.time() - model_start
            print(f"  模型完成: {model_count} 条")
            print(f"  耗时: {model_time:.1f}s ({model_time/model_count*1000:.1f} ms/条)")

            # 保存模型分类日志
            if hasattr(model_classifier, 'save_log'):
                model_classifier.save_log(output_path)

        except Exception as e:
            print(f"  模型加载失败: {e}")
            print("  将未匹配数据归类为 '其他'")

            for i, item, rule_result in needs_model:
                title = item.get('title', '')
                url = item.get('articleLink', item.get('url', ''))

                # 如果规则有结果，使用规则结果
                if rule_result:
                    results.append({
                        'id': i,
                        'title': title,
                        'url': url,
                        'category': rule_result.label,
                        'confidence': rule_result.confidence,
                        'method': 'rule_fallback',
                        'matched_keywords': rule_result.matched_keywords[:5],
                    })
                else:
                    results.append({
                        'id': i,
                        'title': title,
                        'url': url,
                        'category': '其他',
                        'confidence': 0.3,
                        'method': 'fallback',
                    })
    else:
        print("  无需模型处理 (所有数据已被规则匹配)")

    # 按 id 排序
    results.sort(key=lambda x: x['id'])

    # 保存结果
    print("\n[4/4] 保存结果...")
    total_time = time.time() - start_time
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 统计报告
    category_counter = Counter(r['category'] for r in results)
    method_counter = Counter(r['method'] for r in results)

    # 生成统计报告文本
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(" 分类结果统计")
    report_lines.append("=" * 70)
    report_lines.append(f"\n输入文件: {input_path}")
    report_lines.append(f"输出文件: {output_path}")
    report_lines.append(f"\n总计: {len(results)} 条")
    report_lines.append(f"总耗时: {total_time:.1f} 秒")

    report_lines.append("\n分类方法:")
    for method, count in method_counter.most_common():
        method_name = {
            'rule': '规则分类',
            'model': '模型分类',
            'rule_fallback': '规则兜底',
            'fallback': '默认兜底'
        }.get(method, method)
        report_lines.append(f"  {method_name}: {count} ({count/len(results)*100:.1f}%)")

    report_lines.append("\n类别分布:")
    report_lines.append("-" * 50)
    for category, count in category_counter.most_common():
        bar = "█" * int(count / len(results) * 40)
        report_lines.append(f"  {category:4s}: {count:4d} ({count/len(results)*100:5.1f}%) {bar}")

    report_lines.append(f"\n结果已保存至: {output_path}")

    # 打印到控制台
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # 保存统计报告到 txt 文件
    txt_path = output_path.with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"统计报告已保存至: {txt_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="对已清洗的数据进行分类 (多语言规则 + 模型兜底)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_classify.py --start 0 --end 199                              # 处理 id 0-199
  python run_classify.py --start 200 --end 399                            # 处理 id 200-399
  python run_classify.py --model Qwen3-VL-Embedding-8B --start 0 --end 199
  python run_classify.py --model DeepSeek-R1-Distill-Qwen-7B --start 200 --end 399
  python run_classify.py -i data/cleaned/my_data.jsonl                    # 指定输入文件

支持的模型:
  Qwen3-VL-Embedding-8B      - Qwen3 嵌入模型 (默认)
  DeepSeek-R1-Distill-Qwen-7B - DeepSeek 推理模型
        """
    )
    parser.add_argument("-i", "--input", help="输入数据文件路径 (默认: data/cleaned/top400.jsonl)")
    parser.add_argument("-o", "--output", help="输出结果文件路径")
    parser.add_argument("--start", type=int, help="起始 id（包含）")
    parser.add_argument("--end", type=int, help="结束 id（包含）")
    parser.add_argument("--model", default="Qwen3-VL-Embedding-8B",
                       help="模型名称 (默认: Qwen3-VL-Embedding-8B)")
    parser.add_argument("--limit", type=int, help="限制处理条数 (用于测试)")

    args = parser.parse_args()

    # 根据模型名称判断类型
    model_name = args.model
    if "deepseek" in model_name.lower():
        model_type = "deepseek"
    else:
        model_type = "qwen"

    classify(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        model_type=model_type,
        model_path=None,  # 使用默认路径
        start_id=args.start,
        end_id=args.end,
    )


if __name__ == "__main__":
    main()
