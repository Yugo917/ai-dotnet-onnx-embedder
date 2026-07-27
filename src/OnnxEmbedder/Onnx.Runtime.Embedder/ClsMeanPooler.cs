using System.Numerics.Tensors;

namespace Onnx.Runtime.Embedder;

public interface IMeanPooler
{
    /// <summary>
    /// Reduces the model output (Hidden States) into a single embedding vector.
    /// </summary>
    /// <param name="lastHiddenStates">The full buffer of model outputs [SequenceLength * HiddenSize].</param>
    /// <param name="attentionMask">The mask identifying non-padding tokens.</param>
    /// <param name="sequenceLength">Number of tokens in the sequence.</param>
    /// <param name="hiddenSize">The dimensionality of the model (e.g., 384, 768).</param>
    /// <param name="destination">The buffer where the final embedding is written.</param>
    void MeanPool(ReadOnlySpan<float> lastHiddenStates, ReadOnlySpan<long> attentionMask, int sequenceLength, int hiddenSize, Span<float> destination);
}

/// <summary>
/// CLS (Classification) Token Pooling.
/// Extracts the representation of the entire sequence from the first token (the [CLS] token).
/// This is the standard pooling method for BERT-style models.
/// </summary>
public sealed class ClsTokenPooler : IMeanPooler
{
    public void MeanPool(ReadOnlySpan<float> lastHiddenStates, ReadOnlySpan<long> attentionMask, int sequenceLength, int hiddenSize, Span<float> destination)
    {
        // Simply copy the first hidden state (index 0) into the destination span.
        // No allocation is performed thanks to Slice.
        lastHiddenStates.Slice(0, hiddenSize).CopyTo(destination);
    }
}

/// <summary>
/// Mean Pooling with Attention Masking.
/// This is the standard pooling method for SBERT-style models.
/// Calculates the average of all token embeddings while ignoring padding tokens.
/// Formula: $Embedding = \frac{1}{N} \sum_{i=1}^{N} Token\_Vector_i$ where $i$ are valid tokens.
/// </summary>
public sealed class MaskedMeanPooler : IMeanPooler
{
    public void MeanPool(ReadOnlySpan<float> lastHiddenStates, ReadOnlySpan<long> attentionMask, int sequenceLength, int hiddenSize, Span<float> destination)
    {
        destination.Clear(); 
        var validTokenCount = 0f;

        for (var i = 0; i < sequenceLength; i++)
        {
            // Attention Mask: 1 for actual content, 0 for padding.
            if (attentionMask[i] == 1)
            {
                var tokenVector = lastHiddenStates.Slice(i * hiddenSize, hiddenSize);

                // SIMD acceleration via TensorPrimitives to sum vectors efficiently.
                TensorPrimitives.Add(destination, tokenVector, destination);
                validTokenCount++;
            }
        }

        if (validTokenCount > 0)
        {
            // Compute the average by dividing the cumulative sum by the number of valid tokens.
            TensorPrimitives.Divide(destination, validTokenCount, destination);
        }
    }
}