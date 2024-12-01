using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.KernelMemory;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;

namespace KernelMemory.Evaluation.Evaluators;

#pragma warning disable SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
#pragma warning disable SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

public class ContextRecallEvaluator : EvaluationEngine
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

    public async Task<(float Score, IEnumerable<GroundTruthClassification>? Evaluations)> EvaluateAsync(string question, string context, string groundOfTruth)
    {
        var classification = await Try(3, async (remainingTry) =>
        {
            var extraction = await EvaluateContextRecall.InvokeAsync(this.kernel, new KernelArguments
                {
                    { "question", question},
                    { "context", context },
                    { "ground_truth", groundOfTruth }
                }).ConfigureAwait(false);

            return JsonSerializer.Deserialize<GroundTruthClassifications>(extraction.GetValue<string>()!);
        }).ConfigureAwait(false);

        if (classification is null)
        {
            return (0, null);
        }

        return ((float)classification.Evaluations.Count(c => c.Attributed > 0) / (float)classification.Evaluations.Count(), classification.Evaluations);
    }

    public async Task<(float Score, IEnumerable<GroundTruthClassification>? Evaluations)> EvaluateAsync(TestSet.TestSet testSet, MemoryAnswer answer)
    {
        return await EvaluateAsync(testSet.Question,
                                  JsonSerializer.Serialize(answer.RelevantSources.SelectMany(c => c.Partitions.Select(x => x.Text))),
                                  testSet.GroundTruth)
            .ConfigureAwait(false);
    }
}

public class GroundTruthClassifications
{
    [JsonPropertyName("evaluations")]
    public List<GroundTruthClassification> Evaluations { get; set; } = new();
}

public class GroundTruthClassification
{
    public string Reason { get; set; } = null!;

    public int Attributed { get; set; }

    public string PartitionText { get; set; } = null!;
}

#pragma warning restore SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Restore this diagnostic to proceed.
#pragma warning restore SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Restore this diagnostic to proceed.
