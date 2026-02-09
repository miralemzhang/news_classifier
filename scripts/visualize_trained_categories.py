"""
可视化训练得到的类别向量（从 .pt checkpoint 加载）
对比不同训练策略得到的类别中心分布

与 visualize_categories.py 的区别：
- visualize_categories.py: 可视化人工定义的模板向量
- 本脚本: 可视化从标注数据中训练得到的类别平均向量
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

plt.rcParams['axes.unicode_minus'] = False

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

MODELS_DIR = Path("/home/zzh/webpage-classification/models")
OUTPUT_DIR = Path("/home/zzh/webpage-classification/data/eval")


def load_checkpoint(path: str):
    """从 .pt 文件加载训练好的类别嵌入"""
    ckpt = torch.load(path, map_location='cpu')
    categories = ckpt['categories']
    embeddings = {}
    for cat in categories:
        vec = ckpt['category_embeddings'][cat]
        embeddings[cat] = vec.float().numpy()
    return categories, embeddings


def plot_tsne(ax, embeddings_dict, title):
    """在一个 axes 上画 t-SNE 散点图"""
    categories = list(embeddings_dict.keys())
    matrix = np.array([embeddings_dict[c] for c in categories])

    if len(categories) < 5:
        # t-SNE perplexity must be < n_samples
        reducer = PCA(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(categories) - 1))
    coords = reducer.fit_transform(matrix)

    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    for i, cat in enumerate(categories):
        cat_en = CATEGORY_NAMES_EN.get(cat, cat)
        ax.scatter(coords[i, 0], coords[i, 1],
                   c=[colors[i]], s=300, marker='o', edgecolors='black', linewidths=2,
                   zorder=5)
        ax.annotate(cat_en, (coords[i, 0], coords[i, 1]),
                    fontsize=11, ha='center', va='bottom', fontweight='bold',
                    xytext=(0, 12), textcoords='offset points')

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    return coords


def plot_similarity_heatmap(ax, embeddings_dict, title):
    """在一个 axes 上画余弦相似度热力图"""
    categories = list(embeddings_dict.keys())
    matrix = np.array([embeddings_dict[c] for c in categories])

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / norms
    sim = normalized @ normalized.T

    im = ax.imshow(sim, cmap='RdYlBu_r', vmin=0.3, vmax=1.0)

    labels_en = [CATEGORY_NAMES_EN.get(c, c) for c in categories]
    ax.set_xticks(range(len(categories)))
    ax.set_yticks(range(len(categories)))
    ax.set_xticklabels(labels_en, fontsize=9, rotation=45, ha='right')
    ax.set_yticklabels(labels_en, fontsize=9)

    for i in range(len(categories)):
        for j in range(len(categories)):
            color = 'white' if sim[i, j] > 0.65 else 'black'
            ax.text(j, i, f'{sim[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color=color)

    ax.set_title(title, fontsize=12, fontweight='bold')
    return im, sim


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 收集所有可用的 checkpoint（按训练策略分组排列）
    checkpoints = {
        "Simple Avg (n=500)": MODELS_DIR / "matrix_train500.pt",
        "Simple Avg (n=700)": MODELS_DIR / "matrix_train700.pt",
        "Simple Avg (n=800)": MODELS_DIR / "matrix_train800.pt",
        "Contrastive L2": MODELS_DIR / "contrastive_train800_layers2.pt",
        "Contrastive L4": MODELS_DIR / "contrastive_train800_layers4.pt",
        "Hard Neg w2m05": MODELS_DIR / "hard_negative_train800_weight2_margin05.pt",
        "Hard Neg w2m10": MODELS_DIR / "hard_negative_train800_weight2_margin10.pt",
        "Hard Neg w3m05": MODELS_DIR / "hard_negative_train800_weight3_margin05.pt",
    }

    # 只保留存在的文件
    checkpoints = {k: v for k, v in checkpoints.items() if v.exists()}
    if not checkpoints:
        print("未找到任何 .pt checkpoint 文件")
        return

    print(f"找到 {len(checkpoints)} 个训练好的 checkpoint:")
    for name, path in checkpoints.items():
        print(f"  {name}: {path.name}")

    # 加载所有 checkpoint
    all_data = {}
    for name, path in checkpoints.items():
        categories, embeddings = load_checkpoint(str(path))
        all_data[name] = embeddings
        print(f"  加载 {name}: {len(categories)} 类, dim={list(embeddings.values())[0].shape[0]}")

    # =========================================================
    # 图1: 所有模型的 t-SNE 对比 (联合降维)
    # =========================================================
    n_models = len(all_data)
    n_cols = min(4, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes1 = np.array(axes1).flatten()
    # 隐藏多余的子图
    for idx in range(n_models, len(axes1)):
        axes1[idx].set_visible(False)

    # 联合降维：把所有模型的向量放一起做 t-SNE，保证坐标系可比
    all_vectors = []
    all_labels = []
    all_model_ids = []
    categories = list(list(all_data.values())[0].keys())

    for model_idx, (name, emb_dict) in enumerate(all_data.items()):
        for cat in categories:
            all_vectors.append(emb_dict[cat])
            all_labels.append(cat)
            all_model_ids.append(model_idx)

    all_vectors = np.array(all_vectors)
    n_total = len(all_vectors)

    if n_total >= 8:
        reducer = TSNE(n_components=2, random_state=42,
                       perplexity=min(8, n_total - 1))
    else:
        reducer = PCA(n_components=2, random_state=42)

    coords_all = reducer.fit_transform(all_vectors)

    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    cat_color = {cat: colors[i] for i, cat in enumerate(categories)}

    for model_idx, (name, _) in enumerate(all_data.items()):
        ax = axes1[model_idx]
        mask = np.array(all_model_ids) == model_idx
        coords_model = coords_all[mask]
        labels_model = [all_labels[i] for i in range(n_total) if all_model_ids[i] == model_idx]

        for i, cat in enumerate(labels_model):
            cat_en = CATEGORY_NAMES_EN.get(cat, cat)
            ax.scatter(coords_model[i, 0], coords_model[i, 1],
                       c=[cat_color[cat]], s=300, marker='o',
                       edgecolors='black', linewidths=2, zorder=5)
            ax.annotate(cat_en, (coords_model[i, 0], coords_model[i, 1]),
                        fontsize=10, ha='center', va='bottom', fontweight='bold',
                        xytext=(0, 12), textcoords='offset points')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    fig1.suptitle('Trained Category Embeddings — t-SNE (Joint Reduction)',
                  fontsize=15, fontweight='bold', y=1.02)
    fig1.tight_layout()
    tsne_path = OUTPUT_DIR / "trained_category_tsne.png"
    fig1.savefig(tsne_path, dpi=150, bbox_inches='tight')
    print(f"\nt-SNE 图已保存: {tsne_path}")

    # =========================================================
    # 图2: 所有模型的相似度热力图对比
    # =========================================================
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    axes2 = np.array(axes2).flatten()
    for idx in range(n_models, len(axes2)):
        axes2[idx].set_visible(False)

    for idx, (name, emb_dict) in enumerate(all_data.items()):
        im, sim = plot_similarity_heatmap(axes2[idx], emb_dict, name)

    fig2.suptitle('Trained Category Embeddings — Cosine Similarity',
                  fontsize=15, fontweight='bold', y=1.02)
    fig2.tight_layout()
    sim_path = OUTPUT_DIR / "trained_category_similarity.png"
    fig2.savefig(sim_path, dpi=150, bbox_inches='tight')
    print(f"相似度图已保存: {sim_path}")

    # =========================================================
    # 图3: 最优模型的独立大图（Simple Avg train800）
    # =========================================================
    best_name = list(all_data.keys())[0]
    best_emb = all_data[best_name]

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 7))

    # 独立 t-SNE
    categories = list(best_emb.keys())
    matrix = np.array([best_emb[c] for c in categories])
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(matrix)

    for i, cat in enumerate(categories):
        cat_en = CATEGORY_NAMES_EN.get(cat, cat)
        ax3a.scatter(coords_pca[i, 0], coords_pca[i, 1],
                     c=[cat_color[cat]], s=400, marker='o',
                     edgecolors='black', linewidths=2, zorder=5)
        ax3a.annotate(cat_en, (coords_pca[i, 0], coords_pca[i, 1]),
                      fontsize=13, ha='center', va='bottom', fontweight='bold',
                      xytext=(0, 14), textcoords='offset points')

    ax3a.set_title(f'{best_name} — PCA 2D', fontsize=14, fontweight='bold')
    ax3a.set_xlabel('PC1')
    ax3a.set_ylabel('PC2')
    ax3a.grid(True, alpha=0.3)

    # 相似度热力图
    im, sim = plot_similarity_heatmap(ax3b, best_emb, f'{best_name} — Cosine Similarity')
    plt.colorbar(im, ax=ax3b, label='Cosine Similarity', shrink=0.8)

    fig3.suptitle('Trained Category Centers (Learned from 800 Labeled Samples)',
                  fontsize=15, fontweight='bold')
    fig3.tight_layout()
    main_path = OUTPUT_DIR / "trained_category_main.png"
    fig3.savefig(main_path, dpi=150, bbox_inches='tight')
    print(f"主图已保存: {main_path}")

    # =========================================================
    # 打印余弦相似度矩阵
    # =========================================================
    print(f"\n{'='*60}")
    print(f"余弦相似度矩阵 — {best_name}")
    print(f"{'='*60}")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / norms
    sim_matrix = normalized @ normalized.T

    header = f"{'':12s}" + ''.join(f"{CATEGORY_NAMES_EN.get(c,c):12s}" for c in categories)
    print(header)
    for i, c1 in enumerate(categories):
        row = f"{CATEGORY_NAMES_EN.get(c1,c1):12s}"
        row += ''.join(f"{sim_matrix[i,j]:12.4f}" for j in range(len(categories)))
        print(row)

    plt.close('all')
    print(f"\n全部完成！输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
