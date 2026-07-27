# Onnx.Runtime.Embedder

## Purpose

`Onnx.Runtime.Embedder` is a lightweight, model-agnostic C# library engineered for generating high-dimensional dense vector embeddings **100% in-process**, eliminating the need for external server infrastructure or paid Cloud APIs.

It eliminates the typical trade-offs:

* **Cloud APIs (OpenAI, Cohere):** Avoids per-token costs and data privacy risks.
* **External Local Servers (Ollama, vLLM):** Eliminates IPC overhead, complex container orchestration, and high memory footprints.

### Key Capabilities:

* **Zero External Dependencies:** Runs directly inside your .NET process or Docker container for low latency and complete data privacy.
* **Model Agnostic & Flexible Pooling:** Compatible with any Hugging Face ONNX model, featuring native support for various vector dimensions (384, 768, 1024, etc.) and automatic pooling (CLS token, Mean, Masked).
* **Vector DB Ready:** Built-in L2 normalization out of the box, perfectly optimized for cosine similarity and dot-product searches in databases like Qdrant, Chroma, or Milvus.

---

## Get Started

### Install

#### 1. Prerequisites
Download and install the .NET 9.0 SDK:
👉 [https://dotnet.microsoft.com/download/dotnet/9.0](https://dotnet.microsoft.com/download/dotnet/9.0)

#### 2. Model Asset Setup
Download the target ONNX model and tokenizer configuration into the `.assets/` folder:

```bash
mkdir -p .assets/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
curl -L "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/onnx/model.onnx" -o .assets/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/model.onnx
curl -L "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/tokenizer.json" -o .assets/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tokenizer.json
```

---

## Usage

### Quick Start Code Example

```csharp
using Onnx.Runtime.Embedder;

// 1. Configure options
var options = new EmbedderFactoryOptions
{
    ModelName = "paraphrase-multilingual-MiniLM-L12-v2",
    ModelDimension = 384,
    ModelPath = ".assets/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/model.onnx",
    TokenizerType = TokenizerType.FastTokenizer,
    TokenizerPath = ".assets/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tokenizer.json",
    NormalizeVector = true
};

// 2. Instantiate embedder via Bootstrap Factory
var embedder = EmbedderBootstrapFactory.Create(options);

// 3. Generate embeddings
string inputDoc = "Vectorisation de documentation technique et fonctionnelle en C#.";
ReadOnlyMemory<float> embedding = await embedder.Encode(inputDoc);
```

---
