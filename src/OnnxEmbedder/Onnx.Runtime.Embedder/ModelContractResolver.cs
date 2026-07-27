using Microsoft.ML.OnnxRuntime;

namespace Onnx.Runtime.Embedder;

/// <summary>
/// Represents the dynamically resolved ONNX model contract.
/// This describes which inputs and outputs are available in the model,
/// enabling the embedder to auto-adapt to different model architectures.
/// </summary>
public class ModelContract
{
    public required string InputIdsName { get; set; }
    public string? AttentionMaskName { get; set; }
    public string? TokenTypeIdsName { get; set; }
    public string? PositionIdsName { get; set; }
    public required string OutputName { get; set; }
    public required bool OutputIsSentenceLevel { get; set; }
}

/// <summary>
/// Resolves the ONNX model contract by inspecting session metadata.
/// Automatically detects available inputs and selects the appropriate output,
/// enabling "universal" model support without hardcoded names.
/// </summary>
public sealed class ModelContractResolver
{
    /// <summary>
    /// Resolves the model contract from ONNX session metadata.
    /// </summary>
    /// <param name="session">The InferenceSession to introspect.</param>
    /// <returns>A ModelContract describing the model's inputs and outputs.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the model lacks required inputs or outputs.</exception>
    public static ModelContract Resolve(InferenceSession session)
    {
        if (session == null)
            throw new ArgumentNullException(nameof(session));

        var inputNames = ResolveInputs(session);
        var outputName = ResolveOutput(session, out var isSentenceLevel);

        return new ModelContract
        {
            InputIdsName = inputNames.InputIds,
            AttentionMaskName = inputNames.AttentionMask,
            TokenTypeIdsName = inputNames.TokenTypeIds,
            PositionIdsName = inputNames.PositionIds,
            OutputName = outputName,
            OutputIsSentenceLevel = isSentenceLevel
        };
    }

    private static InputMapping ResolveInputs(InferenceSession session)
    {
        var inputNames = session.InputMetadata.Keys.ToHashSet(StringComparer.OrdinalIgnoreCase);

        if (!inputNames.Any(x => x.Equals("input_ids", StringComparison.OrdinalIgnoreCase)))
        {
            throw new InvalidOperationException(
                "ONNX model does not have 'input_ids' input. This is a required input for standard embedding models. " +
                $"Available inputs: {string.Join(", ", inputNames)}");
        }

        var inputIds = inputNames.First(x => x.Equals("input_ids", StringComparison.OrdinalIgnoreCase));
        var attentionMask = inputNames.FirstOrDefault(x => x.Equals("attention_mask", StringComparison.OrdinalIgnoreCase));
        var tokenTypeIds = inputNames.FirstOrDefault(x => x.Equals("token_type_ids", StringComparison.OrdinalIgnoreCase));
        var positionIds = inputNames.FirstOrDefault(x => x.Equals("position_ids", StringComparison.OrdinalIgnoreCase));

        return new InputMapping
        {
            InputIds = inputIds,
            AttentionMask = attentionMask,
            TokenTypeIds = tokenTypeIds,
            PositionIds = positionIds
        };
    }

    private static string ResolveOutput(InferenceSession session, out bool isSentenceLevel)
    {
        var outputs = session.OutputMetadata;

        if (outputs == null || outputs.Count == 0)
        {
            throw new InvalidOperationException("ONNX model has no outputs. The model may be corrupted or incompatible.");
        }

        var outputNames = outputs.Keys.ToList();

        // Priority 1: Look for sentence_embedding (direct output, already pooled)
        var sentenceEmbeddingOutput = outputNames.FirstOrDefault(x =>
            x.Equals("sentence_embedding", StringComparison.OrdinalIgnoreCase));
        if (sentenceEmbeddingOutput != null)
        {
            isSentenceLevel = ValidateSentenceLevelOutput(outputs[sentenceEmbeddingOutput]);
            return sentenceEmbeddingOutput;
        }

        // Priority 2: Look for common sentence-level outputs
        var sentenceLevelCandidates = new[] { "pooler_output", "pooled_output", "cls_output" };
        foreach (var candidate in sentenceLevelCandidates)
        {
            var output = outputNames.FirstOrDefault(x =>
                x.Equals(candidate, StringComparison.OrdinalIgnoreCase));
            if (output != null)
            {
                isSentenceLevel = ValidateSentenceLevelOutput(outputs[output]);
                return output;
            }
        }

        // Priority 3: Look for token-level outputs (will require pooling)
        var tokenLevelCandidates = new[] { "last_hidden_state", "token_embeddings", "hidden_states", "output_0" };
        foreach (var candidate in tokenLevelCandidates)
        {
            var output = outputNames.FirstOrDefault(x =>
                x.Equals(candidate, StringComparison.OrdinalIgnoreCase));
            if (output != null)
            {
                isSentenceLevel = false;
                return output;
            }
        }

        // Fallback: Use first output, attempt to infer from shape
        var firstOutput = outputNames.First();
        var metadata = outputs[firstOutput];
        isSentenceLevel = IsLikelySentenceLevelOutput(metadata);

        return firstOutput;
    }

    private static bool ValidateSentenceLevelOutput(NodeMetadata metadata)
    {
        if (metadata.Dimensions == null || metadata.Dimensions.Length == 0)
        {
            throw new InvalidOperationException(
                "Output has no dimensions. The model may be corrupted or incompatible.");
        }

        // Sentence-level outputs are typically 2D: [batch_size, embedding_dim]
        // The last dimension is the embedding dimension
        if (metadata.Dimensions.Length != 2)
        {
            throw new InvalidOperationException(
                $"Sentence-level output has unexpected shape. Expected 2D [batch, hidden], got {metadata.Dimensions.Length}D with shape [{string.Join(", ", metadata.Dimensions)}].");
        }

        return true;
    }

    private static bool IsLikelySentenceLevelOutput(NodeMetadata metadata)
    {
        if (metadata.Dimensions == null || metadata.Dimensions.Length == 0)
            return false;

        // If 2D, it's sentence-level
        if (metadata.Dimensions.Length == 2)
            return true;

        // If 3D, it's token-level (requires pooling)
        return false;
    }

    private class InputMapping
    {
        public required string InputIds { get; set; }
        public string? AttentionMask { get; set; }
        public string? TokenTypeIds { get; set; }
        public string? PositionIds { get; set; }
    }
}

