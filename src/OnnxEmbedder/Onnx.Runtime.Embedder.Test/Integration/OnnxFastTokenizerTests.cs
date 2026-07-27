using System.ComponentModel;
using System.Text.Json;
using System.Text.Json.Serialization;
using Xunit;

namespace Onnx.Runtime.Embedder.Test.Integration;

[Category("Integration")]
public class OnnxEmbedderFastTokenizerTests
{
    [Fact]
    public async Task OnnxEmbedder_Encode_Normalized_ShouldBeLikePytorchEncoding()
    {
        // Arrange
        var normalizedVector = true;
        var pytorchOutput = CreatePyTorchEncodeResult();
        var notNormalizedPyTorchOutput = pytorchOutput.Where(w => w.Normalized == normalizedVector).ToList();
        var embedder = EmbedderBootstrapFactory.Create(new EmbedderFactoryOptions()
        {
            ModelName = "paraphrase-multilingual-MiniLM-L12-v2",
            ModelDimension = 384,
            ModelPath = Path.Combine(TestHelper.GetGlobalAssetPath(), "models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/model.onnx"),
            TokenizerType = TokenizerType.FastTokenizer,
            TokenizerPath = Path.Combine(TestHelper.GetGlobalAssetPath(), "models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tokenizer.json"),
            NormalizeVector = true
        });
        
        foreach (var expectedRes in notNormalizedPyTorchOutput)
        {
            // Act
            var result = embedder.Encode(expectedRes.Text);
            
            // Assert
            AssertVector.AssertIsNormalizedVector(expectedRes.Vector!);
            AssertVector.AssertCosineSimilarity(expectedRes.Vector!, result.ToArray(), 0.99f);
            AssertVector.AssertDotSimilarity(expectedRes.Vector!, result.ToArray(), 0.99f);
        }
    }
    
    [Fact]
    public async Task OnnxEmbedder_Encode_NotNormalized_ShouldBeLikePytorchEncoding()
    {
        // Arrange
        var normalizedVector = false;
        var pytorchOutput = CreatePyTorchEncodeResult();
        var notNormalizedPyTorchOutput = pytorchOutput.Where(w => w.Normalized == normalizedVector).ToList();
        var embedder = EmbedderBootstrapFactory.Create(new EmbedderFactoryOptions()
        {
            ModelName = "paraphrase-multilingual-MiniLM-L12-v2",
            ModelDimension = 384,
            ModelPath = Path.Combine(TestHelper.GetGlobalAssetPath(), "models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/model.onnx"),
            TokenizerType = TokenizerType.FastTokenizer,
            TokenizerPath = Path.Combine(TestHelper.GetGlobalAssetPath(), "models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tokenizer.json"),
            NormalizeVector = false
        });
        
        foreach (var expectedRes in notNormalizedPyTorchOutput)
        {
            // Act
            var result = embedder.Encode(expectedRes.Text);
            
            // Assert
            AssertVector.AssertIsRawVector(expectedRes.Vector!);
            AssertVector.AssertCosineSimilarity(expectedRes.Vector!, result.ToArray(), 0.99f);
        }
    }

    
    private List<PyTorchEncodeResult> CreatePyTorchEncodeResult()
    {
        // the pytorch output is the type float32 array. in our c# the value will be truncated to float array
        var jsonPyTorchVector = File.ReadAllText("Assets/PyTorch/PyTorchEncodeResult.json");
        var pyTorchEncodedResult = JsonSerializer.Deserialize<PyTorchEncodeResult[]>(jsonPyTorchVector);
        return pyTorchEncodedResult!.ToList();
    }
}


public class PyTorchEncodeResult
{
    [JsonPropertyName("text")]
    public string Text { get; set; }
    [JsonPropertyName("model")]
    public string Model { get; set; }
    [JsonPropertyName("normalized")]
    public bool Normalized { get; set; }
    [JsonPropertyName("dimension")]
    public string Dimension { get; set; }
    [JsonPropertyName("vector")]
    public float[] Vector { get; set; }
}
