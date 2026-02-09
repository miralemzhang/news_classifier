# News Classifier

One of the Intern Projects when working in [GeTui](https://www.getui.com/). Processed over 1 million real HTML data entries using this tool and completed accurate classification.

Webpage classifier using embedding similarity. Mainly powered by [Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) with hard-negative training and fine-tuning.

## How It Works

```
Input (title + text)
        │
        ▼
  Qwen3-VL-Embedding
        │
        ▼
  Text Embedding [4096-dim]
        │
        ▼
  Cosine Similarity with Category Embeddings
        │
        ▼
  Top-2 Category Predictions
```

Category embeddings are pre-computed from labeled training data using hard-negative mining, stored as a 264KB `.pt` matrix.

## Categories

8 categories: Politics, Economy, Military, Society, Technology, Sports, Entertainment, Other

## Performance

| Method | Top-1 Accuracy | Top-2 Recall |
|--------|---------------|-------------|
| Baseline (template) | 85.0% | 96.9% |
| Hard-negative mining | **87.8%** | **97.6%** |

Evaluated on 800 test samples with 6000 training samples.

## Quick Start

### Requirements

- Python 3.10+
- CUDA GPU (16GB+ VRAM)
- ~20GB disk space

### Installation

```bash
pip install torch transformers qwen-vl-utils numpy scikit-learn Pillow
```

### Download Model

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-VL-Embedding-8B --local-dir models/Qwen3-VL-Embedding-8B
```

### Run API Server

```bash
python api/api_server.py --port 8000
```

### API Usage

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "title": "SpaceX launches new satellite",
    "text": "SpaceX successfully launched a new communications satellite into orbit today..."
  }'
```

Response:

```json
{
  "top2": [
    {"label": "科技", "score": 0.8712},
    {"label": "军事", "score": 0.5134}
  ]
}
```

### Train Your Own Category Embeddings

```bash
python src/llm/embedding_classifier.py \
  --data data/labeled/labeled.jsonl \
  --train 6000 \
  --test 800
```

## Project Structure

> **`package/`** is the final production-ready delivery code — a self-contained, deployable classification service. Everything else (`src/`, `scripts/`, `data/`) is the R&D process: experiments, evaluations, and training pipelines.

```
├── api/                         # API server
│   ├── api_server.py            # Entry point
│   ├── test_api.sh              # Test script
│   └── test_request.json        # Sample request
├── src/
│   ├── llm/
│   │   ├── embedding_classifier.py              # Core classifier
│   │   ├── qwen3_vl_embedding.py                # Embedding model wrapper
│   │   ├── qwen_embedding_classifier_hard_negative_mining.py
│   │   ├── qwen_embedding_classifier_contrastive_learning.py
│   │   └── two_stage_classifier.py              # Embedding + Reranker
│   ├── extraction/              # HTML content extraction
│   ├── preprocessing/           # Data cleaning
│   ├── evaluation/              # Evaluation metrics
│   └── utils/                   # Utilities
├── scripts/                     # Utility scripts
│   ├── run_classify.py          # Batch classification runner
│   ├── visualize_categories.py  # Category visualization
│   ├── evaluate.py              # Evaluation script
│   └── sampling.py              # Data sampling
├── package/                     # ** Production-ready delivery code **
│   ├── api_server.py
│   ├── src/llm/
│   └── models/
│       └── hard_negative_train6000.pt           # Pre-trained category embeddings (264KB)
├── configs/                     # Category config & multilingual keywords
└── data/results/                # Experiment logs and evaluation results
```

## License

MIT
