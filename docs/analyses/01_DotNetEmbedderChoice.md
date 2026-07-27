## Technical Specifications

### 🎯 Key Requirements

* **Use Case:** Vectorisationof functional documentation , business documentation and technical documentation.
* **Zero External Dependencies:** 100% in-process inference within C# (no external HTTP servers like Ollama/vLLM, no third-party `.exe`). Everything is self-contained in the DLL/application.
* **Native Multilingual Support:** Seamless handling of French and English.
* **Domain Vocabulary:** Advanced handling of business terminology and domain-specific synonyms.
* **Vector Flexibility:** Independence regarding embedding dimensions and sizing (e.g., 384, 768, 1536 dims).
* **Low Memory Footprint:** Optimized via quantized models (INT8 / FP16).
* **Docker Ready:** Smooth deployment on containerized environments (Linux/Windows).

---

### 🏗 Architecture (.NET)

#### Tech Stack

* **`Microsoft.ML.OnnxRuntime`:** Ultra-fast, in-process local inference engine.

```
[ Business Logic / C# Application ]
                 │
                 ▼
     `IEmbedder`  (.NET Abstraction)
                 │
                 ▼
      `Microsoft.ML.OnnxRuntime + Tokenizers`  (In-Process)
                 │
                 ▼
       [ Quantized ONNX Model ]

```

#### Key Benefits

* **Complete Decoupling:** Business code consumes the standard `IEmbeddingGenerator` interface without tight coupling to a specific AI model.
* **Portability:** Self-contained, lightweight, and deterministic Docker image.