using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.KernelMemory;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;

namespace KernelMemory.Evaluation.Evaluators;

#pragma warning disable SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
#pragma warning disable SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

internal class ContextRecallEvaluator : EvaluationEngine
{
    private readonly Kernel kernel;

    private KernelFunction EvaluateContextRecall => this.kernel.CreateFunctionFromPrompt(GetSKPrompt("Evaluation", "ContextRecall"), new OpenAIPromptExecutionSettings
    {
        Temperature = 1e-8f,
        TopP = 1e-8f,
        Seed = 0,
        ResponseFormat = "json_object",

    }, functionName: nameof(EvaluateContextRecall));

    public ContextRecallEvaluator(Kernel kernel)
    {
        this.kernel = kernel.Clone();
    }

    internal async Task<float> Evaluate(TestSet.TestSet testSet, MemoryAnswer answer, Dictionary<string, object?> metadata)
    {
        var classification = await Try(3, async (remainingTry) =>
        {
            var extraction = await EvaluateContextRecall.InvokeAsync(this.kernel, new KernelArguments
                {
                    { "question", testSet.Question },
                    { "context", JsonSerializer.Serialize(answer.RelevantSources.SelectMany(c => c.Partitions.Select(x => x.Text))) },
                    { "ground_truth", testSet.GroundTruth }
                }).ConfigureAwait(false);

            return JsonSerializer.Deserialize<GroundTruthClassifications>(extraction.GetValue<string>()!);
        }).ConfigureAwait(false);

        if (classification is null)
        {
            return 0;
        }

        metadata.Add($"{nameof(ContextRecallEvaluator)}-Evaluation", classification);

        return (float)classification.Evaluations.Count(c => c.Attributed > 0) / (float)classification.Evaluations.Count();
    }

}

internal class GroundTruthClassifications
{
    [JsonPropertyName("evaluations")]
    public List<GroundTruthClassification> Evaluations { get; set; } = new();
}

internal class GroundTruthClassification
{
    public string Reason { get; set; } = null!;

    public int Attributed { get; set; }

    public string PartitionText { get; set; } = null!;
}

#pragma warning restore SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Restore this diagnostic to proceed.
#pragma warning restore SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Restore this diagnostic to proceed.
