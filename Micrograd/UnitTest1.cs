using FluentAssertions;
using Xunit.Abstractions;

namespace Micrograd;

public class UnitTest1(ITestOutputHelper outputHelper)
{
    [Fact]
    public void Test1()
    {
        var a = Value.Create(-4.0);
        var b = Value.Create(2.0);
        var c = a + b;
        var d = a * b + b.Pow(3);
        c += c + 1;
        c += 1 + c + (-a);
        d += d * 2 + (b + a).ReLU();
        d += 3 * d + (b - a).ReLU();
        var e = c - d;
        var f = e.Pow(2);
        var g = f / 2.0;
        g += 10.0 / f;
        g.Data.Should().BeApproximately(24.7041, 1e-4);
        g.Backward();
        a.Gradient.Should().BeApproximately(138.8338, 1e-4);
        b.Gradient.Should().BeApproximately(645.5773, 1e-4);
    }
}
