# Tech Stack

## Runtime & Framework
- **.NET 9.0** (SDK pinned via `global.json`, `rollForward: latestMinor`, no pre-release)
- C# with `Nullable` and `ImplicitUsings` enabled in all projects

## Key Libraries

| Package | Version | Purpose |
|---|---|---|
| `Microsoft.ML.OnnxRuntime` | 1.25.1 | Native ONNX inference engine (C++ backend) |
| `System.Numerics.Tensors` | 10.0.7 | SIMD-accelerated vector math (`TensorPrimitives`) |
| `Tokenizers.HuggingFace` | 2.21.4 | Rust-based Fast Tokenizer (BPE/Unigram via tokenizer.json) |

## Test Framework

| Package | Version | Purpose |
|---|---|---|
| `xunit` | 2.9.2 | Test framework |
| `xunit.runner.visualstudio` | 2.8.2 | IDE test runner |
| `Microsoft.NET.Test.Sdk` | 17.12.0 | Test host |
| `coverlet.collector` | 6.0.2 | Code coverage |

## Common Commands

All commands run from `src/OnnxEmbedder/` (solution root).

```bash
# Build
dotnet build OnnxEmbedder.sln

# Run all tests
dotnet test OnnxEmbedder.sln

# Run tests for a specific project
dotnet test Onnx.Runtime.Embedder.Test/Onnx.Runtime.Embedder.Test.csproj

# Build in Release mode
dotnet build OnnxEmbedder.sln -c Release
```

## Model Assets

ONNX model files and tokenizer configs live in `.assets/models/` at the repo root. This path is **not** inside `src/` and is resolved at test time via `TestHelper.GetGlobalAssetPath()`, which walks up from the test output directory. Test assets (e.g., PyTorch reference vectors) live in `Onnx.Runtime.Embedder.Test/Assets/` and are copied to output on build.
