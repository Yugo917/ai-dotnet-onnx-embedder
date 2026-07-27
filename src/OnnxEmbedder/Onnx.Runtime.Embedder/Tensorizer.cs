using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Onnx.Runtime.Embedder;

public class TensorCreationOptions
{
    public string TensorNameInputIds { get; set; }
    public string? TensorNameTypeIds { get; set; }
    public string? TensorNameAttentionMask { get; set; }
    public string? TensorNameTokens { get; set; }
    public string? TensorNameWords { get; set; }
    public string? TensorNameOffsets { get; set; }
    public string? TensorNameSpecialTokensMask { get; set; }
    public string? TensorNameOverflowing { get; set; }
}

public interface ITensorizer
{
    public (int sequenceLength, Dictionary<string, DenseTensor<long>> inputs) Tensorize (string text);
    public TokenizerInfo TokenizerInfo { get; }
}

public interface IOnnxTensorizer : ITensorizer
{
    (int sequenceLength, NamedOnnxValue[] onnxInputs) OnnxTensorize(string text);
}

/// <summary>
/// Converts tokenized results into Tensors compatible with the ONNX Runtime.
/// Optimization focus: Minimizing heap allocations and utilizing Span for memory access.
/// </summary>
public sealed class Tensorizer : IOnnxTensorizer
{
    private readonly ITokenizer tokenizer;
    private readonly TensorCreationOptions tensorOptions;

    public Tensorizer(ITokenizer tokenizer, TensorCreationOptions tensorOptions)
    {
        this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        this.tensorOptions = tensorOptions ?? throw new ArgumentNullException(nameof(tensorOptions));
    }
    
    public TokenizerInfo TokenizerInfo { get => this.tokenizer.GetInfo(); }

    /// <summary>
    /// Transforms text into a dictionary of DenseTensors.
    /// AI Concept: Neural networks process data in batches. Even for one sentence, 
    /// we use a shape of [1, SequenceLength], where 1 is the Batch Size.
    /// </summary>
    public (int sequenceLength, Dictionary<string, DenseTensor<long>> inputs) Tensorize(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            throw new ArgumentException("Text cannot be empty", nameof(text));
        }

        var tokenizerResult = tokenizer.Encode(text);
        var seqLen = tokenizerResult.SequenceLength;

        // Optimization: Stack allocation via collection expressions.
        // Using Span avoids heap allocation for the dimensions array.
        ReadOnlySpan<int> dimensions = [1, seqLen];

        var resultTensors = new Dictionary<string, DenseTensor<long>>(4);

        // InputIds: The unique integer ID for each token.
        if (string.IsNullOrEmpty(tensorOptions.TensorNameInputIds))
        {
            throw new InvalidOperationException("TensorNameInputIds is required but not configured.");
        }
        resultTensors.Add(tensorOptions.TensorNameInputIds, CreateTensorFromList(tokenizerResult.InputIds, dimensions));

        // AttentionMask: 1 for real tokens, 0 for padding. 
        // This tells the AI model to ignore the padding during calculations.
        if (tokenizerResult.AttentionMask != null && !string.IsNullOrEmpty(tensorOptions.TensorNameAttentionMask))
        {
            resultTensors.Add(tensorOptions.TensorNameAttentionMask, CreateTensorFromList(tokenizerResult.AttentionMask, dimensions));
        }

        // TypeIds: Used in models like BERT to distinguish between two different sentences.
        if (tokenizerResult.TypeIds != null && !string.IsNullOrEmpty(tensorOptions.TensorNameTypeIds))
        {
            resultTensors.Add(tensorOptions.TensorNameTypeIds, CreateTensorFromList(tokenizerResult.TypeIds, dimensions));
        }

        return (seqLen, resultTensors);
    }

    /// <summary>
    /// Wraps the Tensors into NamedOnnxValue, the format required by the InferenceSession.Run method.
    /// </summary>
    public (int sequenceLength, NamedOnnxValue[] onnxInputs) OnnxTensorize(string text)
    {
        var (seqLen, rawTensors) = Tensorize(text);
        var onnxInputs = new NamedOnnxValue[rawTensors.Count];

        var i = 0;
        foreach (var kvp in rawTensors)
        {
            // External Package Call: Create the ONNX-specific wrapper for the tensor.
            onnxInputs[i++] = NamedOnnxValue.CreateFromTensor(kvp.Key, kvp.Value);
        }

        return (seqLen, onnxInputs);
    }

    /// <summary>
    /// High-performance conversion from IList to DenseTensor.
    /// AI models usually expect 'long' (Int64) for indices, but tokenizers often output 'uint'.
    /// </summary>
    private static DenseTensor<long> CreateTensorFromList(IList<uint> data, ReadOnlySpan<int> dimensions)
    {
        var count = data.Count;
        var destinationArray = new long[count];

        if (data is List<uint> list)
        {
            // Optimization: CollectionsMarshal.AsSpan gives us a direct view into the List's internal array.
            // This avoids the overhead of the indexer (which includes bounds checking for every call).
            var sourceSpan = CollectionsMarshal.AsSpan(list);
            var destSpan = destinationArray.AsSpan();

            for (var i = 0; i < sourceSpan.Length; i++)
            {
                // Manual cast from uint to long for ONNX compatibility.
                destSpan[i] = sourceSpan[i];
            }
        }
        else
        {
            // Fallback for non-list IList implementations.
            for (var i = 0; i < count; i++)
            {
                destinationArray[i] = data[i];
            }
        }

        // External Package Call: Initialize the DenseTensor with our prepared array and dimensions.
        return new DenseTensor<long>(destinationArray, dimensions);
    }
}