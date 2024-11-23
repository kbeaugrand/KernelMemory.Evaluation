
using System.Text.Json;
using Microsoft.KernelMemory;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;

namespace KernelMemory.Evaluation.Evaluators;

#pragma warning disable SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
#pragma warning disable SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

internal class ContextRelevancyEvaluator : EvaluationEngine
{
    private readonly Kernel kernel;

    private KernelFunction EvaluateContext => this.kernel.CreateFunctionFromPrompt(GetSKPrompt("Evaluation", "ContextPrecision"), new OpenAIPromptExecutionSettings
    {
        Temperature = 1e-8f,
        TopP = 1e-8f,
        Seed = 0,
        ResponseFormat = "json_object",

    });

    public ContextRelevancyEvaluator(Kernel kernel)
    {
        this.kernel = kernel.Clone();
    }

    internal async Task<float> Evaluate(MemoryAnswer answer, Dictionary<string, object?> metadata)
    {
        var contextRelevancy = new List<ContextRelevancy>();

        foreach (var item in answer.RelevantSources.SelectMany(c => c.Partitions))
        {
            contextRelevancy.Add(await EvaluateContextRelevancy(item, answer).ConfigureAwait(false));
        }

        metadata.Add($"{nameof(ContextRelevancyEvaluator)}-Evaluation", contextRelevancy);

        return (float)contextRelevancy.Count(c => c.Verdict > 0) / (float)contextRelevancy.Count;
    }

    internal async Task<ContextRelevancy> EvaluateContextRelevancy(Citation.Partition partition, MemoryAnswer answer)
    {
        var relevancy = await Try(3, async (tryCount) =>
        {
            var verification = await EvaluateContext.InvokeAsync(this.kernel, new KernelArguments
                {
                    { "question", answer.Question },
                    { "answer", answer.Result },
                    { "context", partition.Text }
                }).ConfigureAwait(false);

            return JsonSerializer.Deserialize<ContextRelevancy>(verification.GetValue<string>()!);
        }).ConfigureAwait(false);

        relevancy!.PartitionText = partition.Text;

        return relevancy;
    }
}

internal class ContextRelevancy
{
    public string Reason { get; set; } = null!;

    public int Verdict { get; set; }

    public string PartitionText { get; set; } = null!;
}

#pragma warning restore SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Restore this diagnostic to proceed.
#pragma warning restore SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Restore this diagnostic to proceed.
