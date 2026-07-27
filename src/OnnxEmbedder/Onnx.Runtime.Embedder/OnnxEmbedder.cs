using Microsoft.ML.OnnxRuntime.Tensors;

namespace Onnx.Runtime.Embedder;

public interface IEmbedder
{
    ReadOnlyMemory<float> Encode(string inputText);
}

public interface IEmbedderInfo
{
    public EmbedderInfo GetInfo();
}

public class EmbedderInfo
{
    public required ModelInfo ModelInfo { get; set; }
    public required TokenizerInfo TokenizerInfo { get; set; }
}


/// <summary>
/// Represents a high-performance Text Embedder using ONNX Runtime.
/// This class transforms raw strings into mathematical vectors (Embeddings).
/// </summary>
public class OnnxEmbedder : IEmbedder, IEmbedderInfo, IDisposable
{
    private readonly IInferenceEngine inferenceEngine;
    private readonly IOnnxTensorizer tensorizer;
    private readonly IMeanPooler? meanPooler;
    private readonly IVectorNormalizer? normalizer;
    private readonly int vectorDimension;
    private readonly string outputName;
    private readonly bool outputIsSentenceLevel;

    public OnnxEmbedder(
        int vectorDimension,
        IInferenceEngine inferenceEngine,
        IOnnxTensorizer tensorizer,
        string outputName,
        bool outputIsSentenceLevel,
        IMeanPooler? meanPooler = null,
        IVectorNormalizer? normalizer = null
        )
    {
        this.inferenceEngine = inferenceEngine ?? throw new ArgumentNullException(nameof(inferenceEngine));
        this.tensorizer = tensorizer ?? throw new ArgumentNullException(nameof(tensorizer));
        this.meanPooler = meanPooler;
        this.normalizer = normalizer;
        this.outputName = outputName ?? throw new ArgumentNullException(nameof(outputName));
        this.outputIsSentenceLevel = outputIsSentenceLevel;
        
        if (vectorDimension != inferenceEngine.GetInfo().VectorDimension)
        {
            throw new ArgumentException($"Provided vector dimension {vectorDimension} does not match the model's output dimension {inferenceEngine.GetInfo().VectorDimension}");
        }

        this.vectorDimension = vectorDimension;
    }

    /// <summary>
    /// Encodes text into a dense vector.
    /// Logic: Tokenization -> Tensorization -> Inference -> Optional Pooling -> Normalization.
    /// </summary>
    public ReadOnlyMemory<float> Encode(string text)
    {
        // 1. Tokenization: Transforms text into 'InputIds' (numbers) and 'AttentionMask' (importance).
        // This is necessary because Neural Networks only process numerical tensors, not raw text.
        var (sequenceLength, inputs) = tensorizer.OnnxTensorize(text);

        // 2. Inference: Run the ONNX model.
        // The InferenceSession.Run call invokes the native C++ ONNX Runtime engine for maximum hardware acceleration.
        using var results = inferenceEngine.Session.Run(inputs);

        // 3. Extraction: Get the output from the model.
        // This may be a 2D tensor [BatchSize, HiddenDimension] (sentence-level) 
        // or a 3D tensor [BatchSize, SequenceLength, HiddenDimension] (token-level).
        if (results == null || results.Count == 0)
        {
            throw new InvalidOperationException("ONNX inference returned null or empty results.");
        }

        var outputValue = results.FirstOrDefault(x => x.Name == outputName);
        if (outputValue == null)
        {
            throw new InvalidOperationException(
                $"ONNX model did not produce the expected output '{outputName}'. " +
                $"Available outputs: {string.Join(", ", results.Select(r => r.Name))}");
        }

        var denseTensor = outputValue.AsTensor<float>() as DenseTensor<float>;
        if (denseTensor == null)
        {
            throw new InvalidOperationException(
                $"Failed to convert ONNX output '{outputName}' to DenseTensor<float>. Output type: {outputValue.GetType().Name}");
        }

        var embedding = new float[vectorDimension];
        var embeddingSpan = embedding.AsSpan();

        if (outputIsSentenceLevel)
        {
            // Output is already sentence-level [batch, hidden]. Extract the first (and only) row.
            if (denseTensor.Dimensions.Length != 2)
            {
                throw new InvalidOperationException(
                    $"Sentence-level output '{outputName}' has unexpected shape. Expected 2D [batch, hidden], " +
                    $"got {denseTensor.Dimensions.Length}D with shape [{string.Join(", ", denseTensor.Dimensions.ToArray())}].");
            }
            
            denseTensor.Buffer.Span.Slice(0, vectorDimension).CopyTo(embeddingSpan);
        }
        else
        {
            // Output is token-level [batch, seq, hidden]. Apply pooling.
            if (denseTensor.Dimensions.Length != 3)
            {
                throw new InvalidOperationException(
                    $"Token-level output '{outputName}' has unexpected shape. Expected 3D [batch, seq, hidden], " +
                    $"got {denseTensor.Dimensions.Length}D with shape [{string.Join(", ", denseTensor.Dimensions.ToArray())}].");
            }

            if (meanPooler == null)
            {
                throw new InvalidOperationException(
                    "Token-level output requires pooling, but no pooler was provided. " +
                    "Please provide a pooler (e.g., MaskedMeanPooler or ClsTokenPooler).");
            }

            var maskInput = inputs.FirstOrDefault(x => x.Name.Equals("attention_mask", StringComparison.OrdinalIgnoreCase));
            if (maskInput == null)
            {
                throw new InvalidOperationException(
                    "Pooling token-level outputs requires an attention mask, but none was found in the model inputs.");
            }

            var maskValue = maskInput.Value as DenseTensor<long>;
            if (maskValue == null)
            {
                throw new InvalidOperationException("Failed to convert attention mask to DenseTensor<long>.");
            }

            // 4. Mean Pooling: AI models return a vector for every token.
            // Mean Pooling collapses these multiple vectors into a single 'sentence embedding' by averaging them,
            // while ignoring padding tokens defined in the attention mask.
            meanPooler.MeanPool(
                denseTensor.Buffer.Span,
                maskValue.Buffer.Span,
                sequenceLength,
                vectorDimension,
                embeddingSpan
            );
        }

        // 5. Normalization: Normalizes the vector to a unit length (L2 Norm).
        // This is a critical optimization for Vector Databases, allowing the use of Dot Product 
        // as a faster equivalent to Cosine Similarity.
        normalizer?.Normalize(embeddingSpan);

        return embedding;
    }
    
    public EmbedderInfo GetInfo()
    {
        return new EmbedderInfo()
        {
            ModelInfo = inferenceEngine.GetInfo(),
            TokenizerInfo = tensorizer.TokenizerInfo
        };
    }

    public void Dispose()
    {
        inferenceEngine.Session.Dispose();
        GC.SuppressFinalize(this);
    }
}