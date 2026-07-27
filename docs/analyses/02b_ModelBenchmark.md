# Model Benchmark

| Model | Description | Languages | Performance (MTEB) | Size (Params) | Dimensions | Context Window | Optimization | Speed (Latency) | Onnx RAM Usage | GGUF RAM Usage | CPU/GPU | MTEB Score (Ret) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **intfloat/multilingual-e5-small** | The lightweight CPU/Edge champion | Multilingual (100) | High (for its size) | ~118 Million | 384 | 512 tokens | Ultra-low latency | Very fast | ~240 MB (f16) | ~80 MB (Q4) | CPU-optimized (<20ms) | 49.0 |
| **BAAI/bge-m3** | The versatile "Swiss Army Knife" | Multilingual (100+) | Excellent (Hybrid) | ~568 Million | 1024 | **8192 tokens** | Dense, Sparse & ColBERT | Moderate | ~1.1 GB (f16) | ~350 MB (Q4) | GPU recommended | 48.8 (100+ langs) |
| **paraphrase-multilingual-MiniLM-L12-v2** | The classic multilingual standard | Multilingual (50+) | Moderate (baseline) | ~118 Million | 384 | 128 tokens | Paraphrase & Similarity | Very fast | ~240 MB (f16) | ~80 MB (Q4) | CPU / Server | ~40.0 |