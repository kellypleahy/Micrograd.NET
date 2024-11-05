namespace Micrograd;

public class Neuron(int nInputs, bool nonLinear = true)
{
    private readonly List<Value> _weights = Enumerable.Range(0, nInputs)
        .Select(_ => Value.Create(Random.Shared.NextDouble() * 2 - 1.0))
        .ToList();
    private readonly Value _b = Value.Create(0);

    public IEnumerable<Value> Parameters => [.._weights, _b];

    public Value Eval(Value[] x)
    {
        var act = _weights.Zip(x)
            .Select(t => t.First * t.Second)
            .Aggregate((left, right) => left + right) + _b;
        return nonLinear ? act.ReLU() : act;
    }
}