namespace Onnx.Runtime.Embedder.Test;


public static class TestHelper
{
    public static string GetGlobalAssetPath()
    {
        return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "..", ".assets"));
    }
}