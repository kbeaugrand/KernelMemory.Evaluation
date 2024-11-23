namespace KernelMemory.Evaluation.TestSet;

public class TestSet
{
    public string Question { get; set; } = null!;

    public QuestionType QuestionType { get; set; }

    public string GroundTruth { get; set; } = null!;

    public int GroundTruthVerdict { get; set; }

    public IEnumerable<string> Context { get; set; } = Array.Empty<string>();

    public string ContextString => string.Join("\n", this.Context);
}
