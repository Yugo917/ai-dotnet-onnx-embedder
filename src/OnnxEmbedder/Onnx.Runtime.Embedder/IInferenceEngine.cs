using Microsoft.ML.OnnxRuntime;

namespace Onnx.Runtime.Embedder;

public interface IInferenceEngine
{
    public InferenceSession Session { get; }
    public ModelInfo GetInfo();
}

public class ModelInfo
{
    public required string ModelName { get; set; }
    public required string ModelPath { get; set; }
    public int VectorDimension { get; set; }
}

public class InferenceEngine : IInferenceEngine, IDisposable
{
    private readonly string modelName;
    private readonly string modelPath;
    private readonly InferenceSession inferenceSession;
    
    public InferenceEngine(string modelName, string modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Model name cannot be null or empty.", nameof(modelName));
        }

        if (string.IsNullOrWhiteSpace(modelPath))
        {
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        }

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"ONNX model file not found at: {modelPath}", modelPath);
        }

        inferenceSession = new InferenceSession(modelPath);
        this.modelName = modelName;
        this.modelPath = modelPath;
    }

    public InferenceSession Session
    {
        get => this.inferenceSession;
    }
    
    public ModelInfo GetInfo()
    {
        if (inferenceSession.OutputMetadata == null || inferenceSession.OutputMetadata.Count == 0)
        {
            throw new InvalidOperationException("ONNX model has no output metadata. The model may be corrupted or incompatible.");
        }

        var firstOutput = inferenceSession.OutputMetadata.First().Value;
        if (firstOutput.Dimensions == null || firstOutput.Dimensions.Length == 0)
        {
            throw new InvalidOperationException("ONNX model output has no dimensions. The model may be corrupted or incompatible.");
        }

        var vectorDimension = firstOutput.Dimensions.Last();
        if (vectorDimension <= 0)
        {
            throw new InvalidOperationException($"Invalid vector dimension: {vectorDimension}. Expected a positive integer.");
        }

        return new ModelInfo()
        {
            ModelName = modelName,
            ModelPath = modelPath,
            VectorDimension = vectorDimension
        };
    }

    public void Dispose()
    {
        inferenceSession.Dispose();
    }
}