namespace Micrograd;

public class Layer(int nInputs, int nOutputs, bool nonLinear = true)
{
    private readonly List<Neuron> _neurons = Enumerable.Range(0, nOutputs)
        .Select(_ => new Neuron(nInputs, nonLinear))
        .ToList();

    public Value[] Eval(Value[] x) => _neurons.Select(n => n.Eval(x)).ToArray();

    public IEnumerable<Value> Parameters => _neurons.SelectMany(n => n.Parameters);
}