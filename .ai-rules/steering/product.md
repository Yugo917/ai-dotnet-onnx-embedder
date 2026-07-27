# Product: ai-dotnet-onnx-embedder

A .NET library for generating text embeddings locally using ONNX Runtime. It converts raw strings into dense float vectors suitable for use with vector databases and semantic search.

## Core Capability

The pipeline follows this sequence: **Tokenization → Tensorization → ONNX Inference → Pooling (optional) → L2 Normalization (optional)**

The library is designed to be model-agnostic — it auto-discovers the model's input/output contract at runtime via `ModelContractResolver`, meaning it works with any standard HuggingFace-style ONNX embedding model without hardcoded names.

## Key Design Goals

- **Performance-first**: Minimizes heap allocations using `Span<T>`, `CollectionsMarshal.AsSpan`, and SIMD via `TensorPrimitives`
- **Universal model support**: Auto-detects sentence-level vs. token-level outputs and selects the appropriate pooling strategy
- **L2 normalization by default**: Produces unit-length vectors optimized for dot product similarity in vector databases
- **Composable pipeline**: Each stage (tokenizer, tensorizer, pooler, normalizer) is hidden behind an interface and can be swapped independently

## Supported Models

Currently validated against `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions). Models must be in ONNX format with a `tokenizer.json` (HuggingFace Fast Tokenizer format).

## Entry Point

`EmbedderBootstrapFactory.Create(EmbedderFactoryOptions)` is the single public factory for building a configured `IEmbedder` instance.
