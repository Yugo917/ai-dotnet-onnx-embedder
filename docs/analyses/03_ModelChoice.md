# Model Choice

## 1. Evaluation Framework & Expert Recommendations

### Chunking Strategy for Tech Doc & Workflows

* **Business Workflows:** Simple fixed-size chunking (e.g., 512 tokens) breaks the logical flow of a process. Instead, use **hierarchical or structure-based chunking** (headings, BPMN steps, Markdown titles) to keep an entire workflow within a single vector block.
* **Technical Documentation:** Always attach **header metadata** (e.g., `module`, `version`, `type: API/Workflow`) during vectorization to enable pre-filtering (*hybrid search* / *metadata filtering*).

### Selection Criteria for Docker

1. **Low RAM/CPU Footprint:** The selected models consume between **300 MB and 1.5 GB of RAM** during CPU/ONNX inference.
2. **Multilingual Support:** Essential if your business documentation mixes French with technical English.
3. **Inference Optimization:** In Docker environments, prefer using **Text Embeddings Inference (TEI) by Hugging Face**, **Ollama**, or **ONNX Runtime** to minimize latency.

---

## 2. Comparative Table of Lightweight Models (Docker-Ready)

| Model Name | Dimension | Max Tokens | Description & Recommended Usage |
| --- | --- | --- | --- |
| **`intfloat/multilingual-e5-small`** | **384** | **512** | **Best balance between size and performance.** A highly capable multilingual model (~118M params) with strong French performance. *Note: Requires `passage: ` prefix during indexing and `query: ` prefix during retrieval.* |
| **`BAAI/bge-m3`** *(Small / Quantized Version)* | **1024** | **8192** | **Essential for long workflows.** Handles large context windows (8k tokens), multi-vector retrieval (ColBERT), and native dense/sparse search. *Slightly higher RAM footprint (~1.2 GB).* |
| **`nomic-ai/nomic-embed-text-v1.5`** | **768** *(or Matryoshka)* | **8192** | **Ultra-flexible (8k context window).** Supports dynamic dimensioning (*Matryoshka Embeddings*, e.g., truncating down to 256 or 512 without major accuracy loss). Excellent for structured documentation. |
| **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`** | **384** | **128** | **Ultra-lightweight and fast (~470 MB RAM).** Ideal for micro-chunks, business FAQs, or workflow step labels. Limited on complex technical paragraphs. |
| **`Alibaba-NLP/gte-multilingual-base`** | **768** | **8192** | **High precision for domain-specific tasks.** Performs exceptionally well on complex semantic matching and mixed code/documentation. Consumes around 600 MB of RAM. |

---

## 3. Final Recommendation for Your Use Case

> **Our Expert Selection:**
> 1. **Micro & Fast Option:** **`intfloat/multilingual-e5-small`** — If you face strict Docker hardware constraints and your chunks remain under 400 words.
> 2. **Long Workflows Option:** **`BAAI/bge-m3`** or **`nomic-embed-text-v1.5`** — If your business procedures and workflows exceed 500 tokens per step and require keeping full functional context within a single embedding vector.
> 
>