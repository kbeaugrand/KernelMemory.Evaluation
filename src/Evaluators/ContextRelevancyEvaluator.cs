
using System.Text.Json;
using Microsoft.KernelMemory;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;

namespace KernelMemory.Evaluation.Evaluators;

#pragma warning disable SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
#pragma warning disable SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

public class ContextRelevancyEvaluator : EvaluationEngine
{
    private readonly Kernel kernel;

    private KernelFunction EvaluateContext => this.kernel.CreateFunctionFromPrompt(GetSKPrompt("Evaluation", "ContextPrecision"), new OpenAIPromptExecutionSettings
    {
        Temperature = 1e-8f,
        TopP = 1e-8f,
        Seed = 0,
        ResponseFormat = "json_object"
    });

    public ContextRelevancyEvaluator(Kernel kernel)
    {
        this.kernel = kernel.Clone();
    }

    public async Task<(float Score, IEnumerable<ContextRelevancy>? Evaluations)> EvaluateAsync(string question, string answer, IEnumerable<string> partitions)
    {
        var contextRelevancy = new List<ContextRelevancy>();

        foreach (var item in partitions)
        {
            contextRelevancy.Add(await EvaluateContextRelevancy(question, answer, item).ConfigureAwait(false));
        }

        return ((float)contextRelevancy.Count(c => c.Verdict > 0) / (float)contextRelevancy.Count, contextRelevancy);
    }


    public async Task<(float Score, IEnumerable<ContextRelevancy>? Evaluations)> EvaluateAsync(MemoryAnswer answer)
    {
        return await EvaluateAsync(answer.Question, 
                                    answer.Result, 
                                    answer.RelevantSources.SelectMany(c => c.Partitions.Select(x => x.Text)))
            .ConfigureAwait(false);
    }

    internal async Task<ContextRelevancy> EvaluateContextRelevancy(string question, string answer, string context)
    {
        var relevancy = await Try(3, async (tryCount) =>
        {
            var verification = await EvaluateContext.InvokeAsync(this.kernel, new KernelArguments
                {
                    { "question", question },
                    { "answer", answer },
                    { "context", context }
                }).ConfigureAwait(false);

            return JsonSerializer.Deserialize<ContextRelevancy>(verification.GetValue<string>()!);
        }).ConfigureAwait(false);

        relevancy!.PartitionText = context;

        return relevancy;
    }
}

public class ContextRelevancy
{
    public string Reason { get; set; } = null!;

    public int Verdict { get; set; }

    public string PartitionText { get; set; } = null!;
}

#pragma warning restore SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Restore this diagnostic to proceed.
#pragma warning restore SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Restore this diagnostic to proceed.
