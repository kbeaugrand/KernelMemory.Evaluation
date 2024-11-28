
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

    public async Task<(float score, IEnumerable<StatementEvaluation>? evaluations)> EvaluateAsync(string question, string answer, string context)
    {
        var statements = await Try(3, async (remainingTry) =>
        {
            var extraction = await ExtractStatements.InvokeAsync(this.kernel, new KernelArguments
                {
                    { "question", question },
                    { "answer", answer }
                }).ConfigureAwait(false);

            return JsonSerializer.Deserialize<StatementExtraction>(extraction.GetValue<string>()!);
        }).ConfigureAwait(false);

        if (statements is null)
        {
            return (0, null);
        }

        var faithfulness = await Try(3, async (remainingTry) =>
        {
            var evaluation = await FaithfulnessEvaluation.InvokeAsync(this.kernel, new KernelArguments
            {
                    { "context", context },
                    { "answer", answer },
                    { "statements", JsonSerializer.Serialize(statements) }
                }).ConfigureAwait(false);

            var faithfulness = JsonSerializer.Deserialize<FaithfulnessEvaluations>(evaluation.GetValue<string>()!);

            return faithfulness;
        }).ConfigureAwait(false);

        if (faithfulness is null)
        {
            return (0, null);
        }

        return ((float)faithfulness.Evaluations.Count(c => c.Verdict > 0) / (float)statements.Statements.Count(), faithfulness.Evaluations);
    }

    public async Task<(float score, IEnumerable<StatementEvaluation>? evaluations)> EvaluateAsync(MemoryAnswer answer)
    {
        return await EvaluateAsync(
                question: answer.Question,
                answer: answer.Result, 
                context: String.Join(Environment.NewLine, answer.RelevantSources.SelectMany(c => c.Partitions.Select(p => p.Text))))
            .ConfigureAwait(false);
    }
}

public class FaithfulnessEvaluations
{
    [JsonPropertyName("evaluations")]
    public List<StatementEvaluation> Evaluations { get; set; } = new List<StatementEvaluation>();
}

public  class StatementEvaluation
{
    public string Statement { get; set; } = null!;

    public string Reason { get; set; } = null!;

    public int Verdict { get; set; }
}

#pragma warning restore SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Restore this diagnostic to proceed.
