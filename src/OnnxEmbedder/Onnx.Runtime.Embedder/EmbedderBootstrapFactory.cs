namespace Onnx.Runtime.Embedder;

public class EmbedderFactoryOptions
{
    public string ModelName { get; set; }
    public int ModelDimension { get; set; }
    public string ModelPath { get; set; }
    public bool NormalizeVector { get; set; } = true;
    public TokenizerType TokenizerType { get; set; }
    public string TokenizerPath { get; set; }
    public string VocabFilePath { get; set; }
}

public class EmbedderBootstrapFactory
{
    public static IEmbedder Create(EmbedderFactoryOptions options)
    {
        if(options.TokenizerType == TokenizerType.FastTokenizer)
        {
            return CreateOnnxEmbedderWithFastTokenizer(options);
        }
        else if(options.TokenizerType == TokenizerType.WordPieceTokenizer)
        {
            throw new NotImplementedException("WordPieceTokenizer implementation is pending.");
        }
        else
        {
            throw new NotSupportedException($"Tokenizer type {options.TokenizerType} is not supported.");
        }
    }

    private static IEmbedder CreateOnnxEmbedderWithFastTokenizer(EmbedderFactoryOptions options)
    {
        // InferenceEngine with one ONNX Inference Session
        // The InferenceSession is the heaviest object in the pipeline.
        // It manages the native memory for the model weights and the execution graph.
        var inferenceEngine = new InferenceEngine(options.ModelName, options.ModelPath);

        if (options.ModelDimension != inferenceEngine.GetInfo().VectorDimension)
        {
            throw new ArgumentException($"Model dimension mismatch. on RealModel: {inferenceEngine.GetInfo().VectorDimension}, on ProvidedConfig: {options.ModelDimension}");
        }
        
        // Auto-discover the ONNX model contract: which inputs/outputs are available
        var modelContract = ModelContractResolver.Resolve(inferenceEngine.Session);
        
        // Tokenizer Loading the 'tokenizer.json' involves complex parsing of large vocabularies.
        var tokenizerOptions = new TokenizerEncodingOptions
        {
            AddSpecialTokens = true,
            IncludeInputIds = true,
            IncludeTypeIds = modelContract.TokenTypeIdsName != null,
            IncludeAttentionMask = modelContract.AttentionMaskName != null
        };
        var fastTokenizer = new FastTokenizer(tokenizerOptions, options.TokenizerPath);
        
        // Tensor Configuration: Use the dynamically discovered input names from the model contract
        var tensorOptions = new TensorCreationOptions
        {
            TensorNameInputIds = modelContract.InputIdsName,
            TensorNameTypeIds = modelContract.TokenTypeIdsName,
            TensorNameAttentionMask = modelContract.AttentionMaskName
        };
        var tensorizer = new Tensorizer(fastTokenizer, tensorOptions);

        // Pooling strategy: Only needed if the output is token-level (requires pooling)
        IMeanPooler? meanPooler = null;
        if (!modelContract.OutputIsSentenceLevel)
        {
            // Default to masked mean pooling for token-level outputs
            meanPooler = new MaskedMeanPooler();
        }

        // L2VectorNormalizer Adjusts the vector so that its total magnitude equals 1.0.
        var normalizer = options.NormalizeVector ? new L2VectorNormalizer() : null;
        
        return new OnnxEmbedder(
            options.ModelDimension, 
            inferenceEngine, 
            tensorizer, 
            modelContract.OutputName,
            modelContract.OutputIsSentenceLevel,
            meanPooler,
            normalizer
        );
    }
}