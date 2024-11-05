namespace Micrograd;

public class Value
{
    private readonly Value[] _children;
    private string _op;
    private Action _backward;

    public double Data { get; private set; }
    public double Gradient { get; private set; }

    public void ResetGradient() => Gradient = 0;

    private Value(double data, string? op = null, params Value[] children)
    {
        Data = data;
        _op = op ?? data.ToString("F4");
        _children = children;
        _backward = () => { };
    }

    public static Value Create(double data)
    {
        return new Value(data);
    }

    public void Backward()
    {
        // 1. topo walk tree
        var walk = new List<Value>();
        var visited = new HashSet<Value>();

        void Walk(Value value)
        {
            if (!visited.Add(value))
                return;
            foreach (var child in value._children)
                Walk(child);
            walk.Add(value);
        }

        Walk(this);
        walk.Reverse();

        // 2. start with gradient = 1 and do _backward for each step in tree.
        Gradient = 1;
        foreach (var value in walk)
            value._backward();
    }

    public static implicit operator Value(double value) => Create(value);

    private Value Add(Value other)
    {
        var parent = new Value(Data + other.Data, "+", this, other);
        parent._backward = () =>
        {
            Gradient += parent.Gradient;
            other.Gradient += parent.Gradient;
        };
        return parent;
    }

    private Value Mul(Value other)
    {
        var parent = new Value(Data * other.Data, "*", this, other);
        parent._backward = () =>
        {
            Gradient += other.Data * parent.Gradient;
            other.Gradient += Data * parent.Gradient;
        };
        return parent;
    }

    public Value Pow(double power)
    {
        var parent = new Value(Math.Pow(Data, power), $"^{power}", this);
        parent._backward = () =>
        {
            Gradient += power * Math.Pow(Data, power - 1) * parent.Gradient;
        };
        return parent;
    }

    public static Value operator+(Value left, Value right)
    {
        return left.Add(right);
    }

    public static Value operator*(Value left, Value right)
    {
        return left.Mul(right);
    }

    public static Value operator/(Value left, Value right)
    {
        return left.Mul(right.Pow(-1));
    }

    public static Value operator-(Value left, Value right)
    {
        return left.Add(-right);
    }

    public static Value operator-(Value single)
    {
        return single.Mul(Create(-1));
    }

    public override string ToString()
    {
        return $"Value({Data}, {Gradient})";
    }

    public Value ReLU()
    {
        var parent = new Value(Math.Max(0, Data), "ReLU", this);
        parent._backward = () =>
        {
            Gradient += parent.Data > 0 ? parent.Gradient : 0;
        };
        return parent;
    }

    public void Update(double learningRate)
    {
        Data -= learningRate * Gradient;
    }
}
