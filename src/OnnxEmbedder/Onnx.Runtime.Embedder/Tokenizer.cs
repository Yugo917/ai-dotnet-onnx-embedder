using Tokenizers.HuggingFace.Tokenizer;

namespace Onnx.Runtime.Embedder;

public enum TokenizerType
{
    FastTokenizer,
    WordPieceTokenizer
}

public class TokenizerEncodingOptions
{
    public bool AddSpecialTokens { get; set; }
    public bool IncludeInputIds { get; set; }
    public bool IncludeTypeIds { get; set; }
    public bool IncludeTokens { get; set; }
    public bool IncludeWords { get; set; }
    public bool IncludeOffsets { get; set; }
    public bool IncludeSpecialTokensMask { get; set; }
    public bool IncludeAttentionMask { get; set; }
    public bool IncludeOverflowing { get; set; }
}

/// <summary>
/// Represents the output of a tokenization process.
/// In AI, InputIds are the numerical vocabulary indices, 
/// while AttentionMask identifies non-padding tokens.
/// </summary>
public record struct TokenizerResult(
    IList<uint> InputIds,
    IList<uint>? AttentionMask,
    IList<uint>? TypeIds,
    int SequenceLength
);

public interface ITokenizerInfo
{
    public TokenizerInfo GetInfo();
}

public class TokenizerInfo
{
    public TokenizerType TokenizerType { get; set; }
    public string TokenizerPath { get; set; }
    public string VocabFilePath { get; set; }
}

public interface ITokenizer : ITokenizerInfo
{
    TokenizerResult Encode(string text);
}

/// <summary>
/// WordPiece tokenizer implementation.
/// WordPiece is a sub-word tokenization algorithm used primarily by BERT models.
/// It builds a vocabulary by choosing segments that maximize the likelihood of the training data.
/// </summary>
public class WordPieceTokenizer : ITokenizer
{
    private readonly string vocabFilePath;

    public WordPieceTokenizer(string vocabFilePath)
    {
        this.vocabFilePath = vocabFilePath;
    }

    public TokenizerResult Encode(string text)
    {
        // WordPiece typically requires a manual iterative search through the vocabulary.
        throw new NotImplementedException("WordPiece manual implementation is pending.");
    }

    public TokenizerInfo GetInfo()
    {
        return new TokenizerInfo
        {
            TokenizerType = TokenizerType.WordPieceTokenizer,
            VocabFilePath = vocabFilePath
        };
    }
}

/// <summary>
/// Fast tokenizer using the HuggingFace Rust-based 'tokenizers' library.
/// This implementation supports Byte-Pair Encoding (BPE) and Unigram algorithms.
/// BPE (Byte-Pair Encoding) merges the most frequent adjacent pairs of characters/tokens iteratively.
/// </summary>
public class FastTokenizer : ITokenizer
{
    private readonly TokenizerEncodingOptions options;
    private readonly string tokenizerJsonPath;
    private readonly Tokenizer tokenizer;

    public FastTokenizer(TokenizerEncodingOptions options, string tokenizerJsonPath)
    {
        // Optimization: We prevent unsupported configurations early to ensure architectural integrity.
        if (options.IncludeTokens || options.IncludeWords || options.IncludeOffsets || options.IncludeSpecialTokensMask || options.IncludeOverflowing)
        {
            throw new NotSupportedException("This implementation optimized for embeddings does not support metadata (Tokens, Offsets, etc.).");
        }

        this.options = options;
        this.tokenizerJsonPath = tokenizerJsonPath;

        // External Package Call: Load the pre-trained tokenizer configuration (BPE/Unigram) from a JSON file.
        tokenizer = Tokenizer.FromFile(tokenizerJsonPath);
    }

    public TokenizerResult Encode(string text)
    {
        // External Package Call: Perform the actual tokenization. 
        // The HuggingFace library is written in Rust, providing high performance for large batches.
        var encodingList = tokenizer.Encode(
            text,
            addSpecialTokens: options.AddSpecialTokens,
            includeTypeIds: options.IncludeTypeIds,
            includeAttentionMask: options.IncludeAttentionMask,
            includeTokens: options.IncludeTokens,
            includeWords: options.IncludeWords,
            includeOffsets: options.IncludeOffsets,
            includeSpecialTokensMask: options.IncludeSpecialTokensMask,
            includeOverflowing: options.IncludeOverflowing
        );

        var encoding = encodingList.First();

        return new TokenizerResult(
            InputIds: encoding.Ids,
            AttentionMask: options.IncludeAttentionMask ? encoding.AttentionMask : null,
            TypeIds: options.IncludeTypeIds ? encoding.TypeIds : null,
            SequenceLength: encoding.Ids.Count
        );
    }

    public TokenizerInfo GetInfo()
    {
        return new TokenizerInfo
        {
            TokenizerType = TokenizerType.FastTokenizer,
            TokenizerPath = tokenizerJsonPath
        };
    }
}