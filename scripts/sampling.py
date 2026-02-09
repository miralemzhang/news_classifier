"""
分层抽样脚本
从千万级数据中抽取 3000 条用于标注
"""

import json
import random
from pathlib import Path
from collections import defaultdict


# 域名分类规则
DOMAIN_RULES = {
    'central_media': ['xinhuanet.com', 'people.cn', 'cctv.com', 'chinadaily.com.cn'],
    'gov': ['.gov.cn', 'mod.gov.cn'],
    'portal': ['sina.com.cn', 'sohu.com', '163.com', 'qq.com', 'ifeng.com'],
    'forum': ['tieba.baidu.com', 'zhihu.com', 'weibo.com'],
    'tech': ['36kr.com', 'ithome.com', 'cnbeta.com'],
}

# 各层抽样比例
LAYER_RATIOS = {
    'central_media': 0.15,
    'local_media': 0.15,
    'gov': 0.10,
    'portal': 0.25,
    'forum': 0.10,
    'tech': 0.10,
    'other': 0.15
}


def classify_domain(domain: str) -> str:
    """根据域名判断来源类型"""
    for layer, patterns in DOMAIN_RULES.items():
        for pattern in patterns:
            if pattern in domain:
                return layer
    return 'other'


def stratified_sample(input_file: str, output_file: str, n_samples: int = 3000):
    """分层抽样"""
    # 1. 按层分组
    layers = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            domain = data.get('domain', data.get('url', ''))
            layer = classify_domain(domain)
            layers[layer].append(data)
            
            if (i + 1) % 100000 == 0:
                print(f"已处理 {i + 1} 条...")
    
    print(f"\n各层数据量:")
    for layer, items in layers.items():
        print(f"  {layer}: {len(items)}")
    
    # 2. 各层抽样
    sampled = []
    for layer, ratio in LAYER_RATIOS.items():
        n = int(n_samples * ratio)
        pool = layers.get(layer, [])
        
        if len(pool) >= n:
            selected = random.sample(pool, n)
        else:
            selected = pool  # 不够就全取
            print(f"警告: {layer} 层样本不足，仅 {len(pool)} 条")
        
        sampled.extend(selected)
        print(f"{layer}: 抽取 {len(selected)} 条")
    
    # 3. 打乱并保存
    random.shuffle(sampled)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n抽样完成，共 {len(sampled)} 条，保存至: {output_file}")


def split_dataset(input_file: str, output_dir: str, 
                  train_ratio=0.7, dev_ratio=0.1, test_ratio=0.2):
    """划分训练/开发/测试集"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    random.shuffle(data)
    n = len(data)
    
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)
    
    train_data = data[:train_end]
    dev_data = data[train_end:dev_end]
    test_data = data[dev_end:]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, subset in [('train', train_data), ('dev', dev_data), ('test', test_data)]:
        with open(output_path / f'{name}.jsonl', 'w', encoding='utf-8') as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"{name}: {len(subset)} 条")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage:")
        print("  抽样: python sampling.py sample <input.jsonl> <output.jsonl> [n_samples]")
        print("  划分: python sampling.py split <input.jsonl> <output_dir>")
        sys.exit(1)
    
    if sys.argv[1] == 'sample':
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
        stratified_sample(sys.argv[2], sys.argv[3], n)
    elif sys.argv[1] == 'split':
        split_dataset(sys.argv[2], sys.argv[3])
