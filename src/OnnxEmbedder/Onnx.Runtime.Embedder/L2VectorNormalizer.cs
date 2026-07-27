using System.Numerics.Tensors;

namespace Onnx.Runtime.Embedder;

public interface IVectorNormalizer
{
    /// <summary>
    /// Adjusts the vector so that its total magnitude equals 1.0.
    /// </summary>
    void Normalize(Span<float> vector);
}

/// <summary>
/// Implements L2 Normalization (Euclidean Norm).
/// AI Concept: This projects the vector onto a "unit hypersphere." 
/// After normalization, the distance between vectors can be calculated using simpler geometry.
/// Formula: $v_{normalized} = \frac{v}{\sqrt{\sum v_i^2}}$
/// </summary>
public sealed class L2VectorNormalizer : IVectorNormalizer
{
    public void Normalize(Span<float> vector)
    {
        if (vector.IsEmpty)
        {
            return;
        }

        // External Package Call: TensorPrimitives.Norm utilizes hardware-specific SIMD 
        // instructions (like AVX-512) to calculate the square root of the sum of squares.
        var norm = TensorPrimitives.Norm(vector);

        // Optimization: We check against float.Epsilon (the smallest positive value > 0)
        // to prevent "Division by Zero" errors which result in NaN vectors.
        if (norm <= float.Epsilon)
        {
            return;
        }

        // Scaling the vector: Divide every element by the norm.
        // Result: The new magnitude of the vector will be exactly 1.0.
        TensorPrimitives.Divide(vector, norm, vector);
    }
}