#!/usr/bin/env python3
"""
网页分类 HTTP API 服务

启动:
    python api_server.py
    python api_server.py --port 8000

请求示例:
    curl -X POST http://localhost:8000/classify \
        -H "Content-Type: application/json" \
        -d '{"articleLink": "https://example.com/news", "title": "央行宣布降息", "text": "中国人民银行宣布..."}'

返回示例:
    {
        "top2": [
            {"label": "经济", "score": 0.87},
            {"label": "时政", "score": 0.71}
        ]
    }
"""

import sys
import json
import argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 全局分类器实例（启动时加载）
classifier = None


def load_model(matrix_path: str = None, model_path: str = None):
    """加载嵌入模型和类别矩阵"""
    from src.llm.embedding_classifier import QwenEmbeddingClassifier

    if matrix_path is None:
        matrix_path = str(PROJECT_ROOT / "models" / "hard_negative_train6000.pt")
    if model_path is None:
        model_path = str(PROJECT_ROOT / "models" / "Qwen3-VL-Embedding-8B")

    clf = QwenEmbeddingClassifier(model_path=model_path, use_template=False)
    clf.load_category_matrix(matrix_path)
    return clf


class ClassifyHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器"""

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def do_POST(self):
        if self.path != "/classify":
            self._send_json({"error": "Not found. Use POST /classify"}, 404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json({"error": "Empty request body"}, 400)
            return

        body = self.rfile.read(content_length)
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        title = req.get("title", "")
        text = req.get("text", "")
        article_link = req.get("articleLink", "")

        if not title and not text:
            self._send_json({"error": "At least one of 'title' or 'text' is required"}, 400)
            return

        # 分类
        result = classifier.classify(title=title, url=article_link, content=text)

        # 取 top 2
        sorted_scores = sorted(result.scores.items(), key=lambda x: -x[1])
        top2 = [{"label": label, "score": round(score, 4)} for label, score in sorted_scores[:2]]

        self._send_json({"top2": top2})

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "Use POST /classify or GET /health"}, 404)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="网页分类 API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="监听端口 (默认: 8000)")
    parser.add_argument("--matrix", help="类别嵌入矩阵路径 (默认: models/hard_negative_train6000.pt)")
    parser.add_argument("--model", help="嵌入模型路径 (默认: models/Qwen3-VL-Embedding-8B)")
    args = parser.parse_args()

    global classifier
    print("正在加载模型...")
    classifier = load_model(matrix_path=args.matrix, model_path=args.model)
    print("模型加载完成!")

    server = HTTPServer((args.host, args.port), ClassifyHandler)
    print(f"\nAPI 服务已启动: http://{args.host}:{args.port}")
    print(f"  POST /classify  - 分类接口")
    print(f"  GET  /health    - 健康检查")
    print(f"\n示例:")
    print(f'  curl -X POST http://localhost:{args.port}/classify \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"title": "央行宣布降息", "text": "中国人民银行宣布...", "articleLink": "https://example.com"}}\'')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.server_close()


if __name__ == "__main__":
    main()
