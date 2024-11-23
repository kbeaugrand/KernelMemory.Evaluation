
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.KernelMemory;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;

namespace KernelMemory.Evaluation.Evaluators;

#pragma warning disable SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

public class FaithfulnessEvaluator : EvaluationEngine
{
    private readonly Kernel kernel;

    private KernelFunction ExtractStatements => this.kernel.CreateFunctionFromPrompt(this.GetSKPrompt("Extraction", "Statements"), new OpenAIPromptExecutionSettings
    {
        Temperature = 1e-8f,
        TopP = 1e-8f,
        Seed = 0,
        ResponseFormat = "json_object",
    }, functionName: nameof(ExtractStatements));

    private KernelFunction FaithfulnessEvaluation => this.kernel.CreateFunctionFromPrompt(this.GetSKPrompt("Evaluation", "Faithfulness"), new OpenAIPromptExecutionSettings
    {
        Temperature = 1e-8f,
        TopP = 1e-8f,
        Seed = 0,
        ResponseFormat = "json_object",
    }, functionName: nameof(FaithfulnessEvaluation));

    public FaithfulnessEvaluator(Kernel kernel)
    {
        this.kernel = kernel.Clone();
    }

    public async Task<float> Evaluate(MemoryAnswer answer, Dictionary<string, object?> metadata)
    {
        var statements = await Try(3, async (remainingTry) =>
        {
            var extraction = await ExtractStatements.InvokeAsync(this.kernel, new KernelArguments
                {
                    { "question", answer.Question },
                    { "answer", answer.Result }
                }).ConfigureAwait(false);

            return JsonSerializer.Deserialize<StatementExtraction>(extraction.GetValue<string>()!);
        }).ConfigureAwait(false);

        if (statements is null)
        {
            return 0;
        }

        var faithfulness = await Try(3, async (remainingTry) =>
        {
            var evaluation = await FaithfulnessEvaluation.InvokeAsync(this.kernel, new KernelArguments
            {
                    { "context", String.Join(Environment.NewLine, answer.RelevantSources.SelectMany(c => c.Partitions.Select(p => p.Text))) },
                    { "answer", answer.Result },
                    { "statements", JsonSerializer.Serialize(statements) }
                }).ConfigureAwait(false);

            var faithfulness = JsonSerializer.Deserialize<FaithfulnessEvaluations>(evaluation.GetValue<string>()!);

            return faithfulness;
        }).ConfigureAwait(false);

        if (faithfulness is null)
        {
            return 0;
        }

        metadata.Add($"{nameof(FaithfulnessEvaluator)}-Evaluation", faithfulness.Evaluations);

        return (float)faithfulness.Evaluations.Count(c => c.Verdict > 0) / (float)statements.Statements.Count();
    }
}

internal class FaithfulnessEvaluations
{
    [JsonPropertyName("evaluations")]
    public List<StatementEvaluation> Evaluations { get; set; } = new List<StatementEvaluation>();
}

internal class StatementEvaluation
{
    public string Statement { get; set; } = null!;

    public string Reason { get; set; } = null!;

    public int Verdict { get; set; }
}

#pragma warning restore SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Restore this diagnostic to proceed.
