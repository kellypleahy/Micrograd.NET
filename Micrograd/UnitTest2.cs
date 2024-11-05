using System.Diagnostics.CodeAnalysis;
using FluentAssertions;
using Xunit.Abstractions;

namespace Micrograd;

public class UnitTest2(ITestOutputHelper outputHelper)
{
    [Fact]
    [SuppressMessage("ReSharper", "InconsistentNaming")]
    public void Test1()
    {
        var model = new MLP(2, 16, 16, 1);
        model.Parameters.Count().Should().Be(337);

        var X = LoadCsv("x.csv", 2);
        var y = LoadCsv("y.csv", 1);

        (Value loss, double accuracy) Loss()
        {
            var scores = X.Select(model.Eval).ToArray();

            // max-margin loss
            var losses = y.Zip(scores, (yi, si) => (1 - yi[0] * si[0]).ReLU()).ToArray();
            var dataLoss = losses.Aggregate((a, b) => a + b) * (1.0 / losses.Length);

            // L2 regularization
            var alpha = 1e-4;
            var regLoss = alpha * model.Parameters.Select(x => x * x).Aggregate((a, b) => a + b);
            var totalLoss = dataLoss + regLoss;

            // accuracy
            var accuracy = y.Zip(scores, (yi, si) => Math.Sign(yi[0].Data) == Math.Sign(si[0].Data))
                .ToArray();

            return (totalLoss, accuracy.Count(x => x) / (1.0 * accuracy.Length));
        }

        var (l, a) = Loss();
        outputHelper.WriteLine($"{l} {a}");

        // optimization
        for (var k = 0; k < 100; k++)
        {
            (l, a) = Loss();
            model.ZeroGradient();
            l.Backward();

            var learningRate = 1.0 - 0.9 * k / 100.0;
            foreach (var p in model.Parameters)
                p.Update(learningRate);

            outputHelper.WriteLine($"step {k} loss {l.Data}, accuracy {a*100:0.00}%");
        }

        l.Data.Should().BeApproximately(0.02, 0.02);
        a.Should().BeApproximately(1.0, 0.02);
    }

    private static Value[][] LoadCsv(string name, int columns)
    {
        var path = Path.Combine(Directory.GetCurrentDirectory(), "data", name);
        var lines = File.ReadAllLines(path);
        return lines.Select(line => line.Split(',', columns))
            .Where(line => line.Length == columns)
            .Select(line => line.Select(double.Parse).Select(Value.Create).ToArray())
            .ToArray();
    }
}
