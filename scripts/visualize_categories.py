"""
可视化 CATEGORY_TEMPLATES 的8个类别向量
使用 t-SNE/PCA 将 3072 维降到 2 维
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 设置字体
plt.rcParams['axes.unicode_minus'] = False

# 中英文类别名映射
CATEGORY_NAMES_EN = {
    "时政": "Politics",
    "经济": "Economy",
    "军事": "Military",
    "社会": "Society",
    "科技": "Tech",
    "体育": "Sports",
    "娱乐": "Entertainment",
    "其他": "Other",
}

# 类别模板 - 精简版（每类3条，与 embedding_classifier.py 保持同步）
CATEGORY_TEMPLATES = {
    "时政": [
        "国家领导人会议讲话，政府工作报告，政策法规发布",
        "两会新闻，党建工作，政府机构改革，干部任免，反腐倡廉",
        "选举投票，议会立法，政党政治，政府换届，政治选举",
        "外交访问，国际关系声明，政府间会谈，外交政策",
        "政府预算案讨论，财政政策辩论，议会预算审议",
    ],
    "经济": [
        "股票市场行情，基金理财，银行利率调整，金融市场波动",
        "企业财报，上市公司，投资并购，公司业绩，商业新闻",
        "房价走势，物价指数，GDP增长数据，经济增长率统计",
        "贸易进出口数据，国际贸易额，进出口统计",
        "消费零售，电商平台，金融产品，创业融资，市场分析",
    ],
    "军事": [
        "军队演习，武器装备，国防建设",
        "战争冲突，军事行动，武装冲突，军事打击，战场战况",
        "军舰航母，战斗机，导弹，坦克，军事科技，武器系统",
        "空袭轰炸，炮击袭击，导弹袭击，军事干预，军事打击行动",
        "军事部署，边境武装对峙，前线战况，部队调动",
    ],
    "社会": [
        "民生新闻，社会热点，公共事件，市民生活",
        "教育医疗，交通出行，社会保障，公共服务",
        "列车事故，车祸，空难，沉船，工业事故，安全事故",
        "犯罪案件，社会治安，法律诉讼，民事纠纷，刑事案件",
        "人道主义危机，饥荒，疫病，难民生存状况",
    ],
    "科技": [
        "人工智能，互联网科技，手机数码产品，科技公司",
        "科学研究，技术创新，航天航空，太空探索",
        "新能源汽车，芯片半导体，5G通信，电动车",
        "软件应用，云计算，大数据，网络安全",
        "机器人，无人机技术研发，智能设备，科技产品发布",
    ],
    "体育": [
        "足球篮球比赛，运动员，体育赛事，比赛结果",
        "奥运会，世界杯，NBA，CBA联赛，欧冠",
        "健身运动，马拉松，电子竞技，极限运动",
        "网球高尔夫，游泳田径，冬季运动",
        "体育明星，球队转会，体育评论，赛事分析",
    ],
    "娱乐": [
        "明星八卦，影视综艺，音乐演唱会，娱乐新闻",
        "电影电视剧，选秀节目，网红直播",
        "游戏动漫，时尚潮流，颁奖典礼",
        "娱乐圈，粉丝文化，偶像团体",
        "艺术展览，文艺演出，戏剧话剧，文化节目",
    ],
    "其他": [
        "生活百科，美食旅游，健康养生",
        "天气预报，星座运势，历史文化",
        "日常杂谈，情感故事，职场经验，个人博客",
        "宠物动物，家居装修，购物指南，生活服务",
        "广告软文，营销推广，品牌宣传",
    ],
}


def load_embeddings(model_path: str = None):
    """加载模型并计算类别嵌入"""
    import sys
    sys.path.insert(0, '/home/zzh/webpage-classification')
    from src.llm.qwen3_vl_embedding import Qwen3VLEmbedder

    if model_path is None:
        model_path = "/home/zzh/webpage-classification/models/Qwen3-VL-Embedding-8B"

    print(f"加载模型: {model_path}")
    embedder = Qwen3VLEmbedder(
        model_name_or_path=model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    category_embeddings = {}
    all_template_embeddings = {}  # 保存每个模板的单独嵌入

    print("计算类别嵌入...")
    for category, templates in CATEGORY_TEMPLATES.items():
        inputs = [{"text": t, "instruction": "表示这段文本的类别特征。"} for t in templates]
        embeddings = embedder.process(inputs)

        # 保存每个模板的嵌入 (转换为 float32 以兼容 numpy)
        all_template_embeddings[category] = embeddings.cpu().float().numpy()

        # 取平均作为类别嵌入
        category_embeddings[category] = embeddings.mean(dim=0).cpu().float().numpy()
        print(f"  {category}: {len(templates)} 个模板 → 向量维度 {embeddings.shape[1]}")

    return category_embeddings, all_template_embeddings


def visualize_categories(category_embeddings, all_template_embeddings, output_path: str):
    """可视化类别向量"""

    categories = list(category_embeddings.keys())
    embeddings_matrix = np.array([category_embeddings[c] for c in categories])

    print(f"\n类别向量矩阵形状: {embeddings_matrix.shape}")
    print(f"共 {len(categories)} 个类别，每个向量 {embeddings_matrix.shape[1]} 维")

    # 收集所有模板嵌入用于 t-SNE
    all_templates = []
    template_labels = []
    for cat, embs in all_template_embeddings.items():
        for emb in embs:
            all_templates.append(emb)
            template_labels.append(cat)
    all_templates = np.array(all_templates)

    # 合并类别中心和所有模板
    combined = np.vstack([embeddings_matrix, all_templates])

    # 使用 t-SNE 降维
    print("\n正在使用 t-SNE 降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined)-1))
    combined_2d = tsne.fit_transform(combined)

    # 分离类别中心和模板点
    centers_2d = combined_2d[:len(categories)]
    templates_2d = combined_2d[len(categories):]

    # 颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    category_colors = {cat: colors[i] for i, cat in enumerate(categories)}

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ========== 图1: 只显示8个类别中心 ==========
    ax1 = axes[0]
    for i, cat in enumerate(categories):
        cat_en = CATEGORY_NAMES_EN.get(cat, cat)
        ax1.scatter(centers_2d[i, 0], centers_2d[i, 1],
                   c=[category_colors[cat]], s=300, marker='o', edgecolors='black', linewidths=2)
        ax1.annotate(cat_en, (centers_2d[i, 0], centers_2d[i, 1]),
                    fontsize=14, ha='center', va='bottom', fontweight='bold',
                    xytext=(0, 10), textcoords='offset points')

    ax1.set_title('8 Category Centers (t-SNE)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.grid(True, alpha=0.3)

    # ========== 图2: 显示类别中心 + 所有模板点 ==========
    ax2 = axes[1]

    # 先画模板点（小点）
    for i, (x, y) in enumerate(templates_2d):
        cat = template_labels[i]
        ax2.scatter(x, y, c=[category_colors[cat]], s=50, alpha=0.5, marker='o')

    # 再画类别中心（大点）
    for i, cat in enumerate(categories):
        cat_en = CATEGORY_NAMES_EN.get(cat, cat)
        ax2.scatter(centers_2d[i, 0], centers_2d[i, 1],
                   c=[category_colors[cat]], s=300, marker='*', edgecolors='black', linewidths=2)
        ax2.annotate(cat_en, (centers_2d[i, 0], centers_2d[i, 1]),
                    fontsize=12, ha='center', va='bottom', fontweight='bold',
                    xytext=(0, 10), textcoords='offset points')

    ax2.set_title('Category Centers + Template Points (t-SNE)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.grid(True, alpha=0.3)

    # 添加图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=category_colors[cat], markersize=10,
                                   label=CATEGORY_NAMES_EN.get(cat, cat))
                      for cat in categories]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化已保存到: {output_path}")

    # ========== 额外：计算类别之间的余弦相似度 ==========
    print("\n" + "="*60)
    print("类别之间的余弦相似度矩阵:")
    print("="*60)

    # 归一化
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    normalized = embeddings_matrix / norms

    # 计算余弦相似度
    similarity_matrix = np.dot(normalized, normalized.T)

    # 打印矩阵
    print(f"\n{'':8s}", end='')
    for cat in categories:
        print(f"{cat:8s}", end='')
    print()

    for i, cat1 in enumerate(categories):
        print(f"{cat1:8s}", end='')
        for j, cat2 in enumerate(categories):
            print(f"{similarity_matrix[i,j]:8.3f}", end='')
        print()

    # 保存相似度热力图
    fig2, ax3 = plt.subplots(figsize=(10, 8))
    im = ax3.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)

    categories_en = [CATEGORY_NAMES_EN.get(c, c) for c in categories]
    ax3.set_xticks(range(len(categories)))
    ax3.set_yticks(range(len(categories)))
    ax3.set_xticklabels(categories_en, fontsize=12)
    ax3.set_yticklabels(categories_en, fontsize=12)

    # 在每个格子里显示数值
    for i in range(len(categories)):
        for j in range(len(categories)):
            text = ax3.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                           ha='center', va='center', fontsize=10,
                           color='white' if similarity_matrix[i, j] > 0.5 else 'black')

    ax3.set_title('Category Cosine Similarity Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Cosine Similarity')

    similarity_path = output_path.replace('.png', '_similarity.png')
    plt.savefig(similarity_path, dpi=150, bbox_inches='tight')
    print(f"相似度矩阵已保存到: {similarity_path}")

    plt.close('all')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--output', type=str, default='/home/zzh/webpage-classification/data/eval/category_embeddings.png')
    args = parser.parse_args()

    category_embeddings, all_template_embeddings = load_embeddings(args.model)
    visualize_categories(category_embeddings, all_template_embeddings, args.output)
