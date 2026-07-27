using System.Numerics.Tensors;
using Xunit;

namespace Onnx.Runtime.Embedder.Test;

public class AssertVector
{
    /// <summary>
    /// Asserts that a given vector is normalized, meaning its L2 norm (Euclidean length) is approximately 1.
    /// </summary>
    /// <param name="vector"></param>
    /// <param name="epsilon"></param>
    public static void AssertIsNormalizedVector(float[] vector, float epsilon = 0.00001f)
    {
        Assert.NotNull(vector);
        Assert.NotEmpty(vector);

        // Optimized L2 norm calculation
        var norm = TensorPrimitives.Norm(vector);

        // Use epsilon tolerance for floating-point comparison.
        // xUnit has precision overloads, otherwise we perform the manual calculation.
        Assert.True(Math.Abs(norm - 1.0f) <= epsilon, 
            $"The vector is not normalized. Expected norm: 1.0, Actual norm: {norm} (Epsilon: {epsilon}).");
    }

    /// <summary>
    /// Asserts that a given vector is not normalized, meaning its L2 norm (Euclidean length) is significantly different from 1.
    /// </summary>
    /// <param name="vector"></param>
    public static void AssertIsRawVector(float[] vector)
    {
        Assert.NotNull(vector);
        var norm = TensorPrimitives.Norm(vector);

        // We assert that it is FALSE that the norm is close to 1.0.
        Assert.False(Math.Abs(norm - 1.0f) < 0.01f, 
            $"The vector appears to be already normalized (Actual norm: {norm}). " +
            $"Expected a raw vector with a norm significantly different from 1.0.");
    }

    /// <summary>
    /// Asserts that the cosine similarity between two vectors meets or exceeds a specified threshold.
    /// </summary>
    /// <param name="expected"></param>
    /// <param name="actual"></param>
    /// <param name="similarityThreshold"></param>
    public static void AssertCosineSimilarity(float[] expected, float[] actual, double similarityThreshold)
    {
        var similarity = TensorPrimitives.CosineSimilarity(expected, actual);
        Assert.True(similarity >= similarityThreshold, $"String similarity {similarity} is below the threshold {similarityThreshold}.");
    }

    /// <summary>
    /// Asserts that the Dot Product similarity between two vectors meets or exceeds a specified threshold.
    /// Important: Both vectors must be normalized (L2 Norm = 1) for this to be a valid semantic similarity measure.
    /// </summary>
    /// <param name="expected"></param>
    /// <param name="actual"></param>
    /// <param name="similarityThreshold"></param>
    public static void AssertDotSimilarity(float[] expected, float[] actual, double similarityThreshold)
    {
        AssertVector.AssertIsNormalizedVector(expected);
        AssertVector.AssertIsNormalizedVector(actual);
    
        // Calculate the Dot Product
        var dotProduct = TensorPrimitives.Dot(expected, actual);

        Assert.True(dotProduct >= (float)similarityThreshold, 
            $"Dot similarity {dotProduct} is below the threshold {similarityThreshold}.");
    }
}