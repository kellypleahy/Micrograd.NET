namespace Micrograd;

public class MLP
{
    private readonly List<Layer> _layers;

    public MLP(int nIn, params int[] nOutputs)
    {
        int[] sizes = [nIn, ..nOutputs];
        var n = nOutputs.Length;
        _layers = Enumerable.Range(0, n - 1)
            .Select(i => new Layer(sizes[i], sizes[i + 1]))
            .Concat([new Layer(sizes[n - 1], sizes[n], false)])
            .ToList();
    }

    public Value[] Eval(Value[] x) => _layers.Aggregate(x, (a, layer) => layer.Eval(a));

    public IEnumerable<Value> Parameters => _layers.SelectMany(l => l.Parameters);

    public void ZeroGradient()
    {
        foreach (var p in Parameters)
            p.ResetGradient();
    }
}