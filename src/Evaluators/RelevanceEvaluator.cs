using System.Numerics.Tensors;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.KernelMemory;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Embeddings;

namespace KernelMemory.Evaluation.Evaluators;
#pragma warning disable SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
#pragma warning disable SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

public class RelevanceEvaluator : EvaluationEngine
{
    private readonly Kernel kernel;

    private ITextEmbeddingGenerationService textEmbeddingGenerationService;

    private KernelFunction ExtractQuestion => this.kernel.CreateFunctionFromPrompt(GetSKPrompt("Extraction", "Question"), new OpenAIPromptExecutionSettings
    {
        Temperature = 1e-8f,
        TopP = 1e-8f,
        Seed = 0,
        ResponseFormat = "json_object",
    }, functionName: nameof(ExtractQuestion));

    public RelevanceEvaluator(Kernel kernel)
    {
        this.kernel = kernel.Clone();

        this.textEmbeddingGenerationService = this.kernel.Services.GetRequiredService<ITextEmbeddingGenerationService>();
    }

    public async Task<(float Score, IEnumerable<(RelevanceEvaluation, float)> Evaluations)> EvaluateAsync(string question, string answer, string context, int strictness = 3)
    {
        var questionEmbeddings = await this.textEmbeddingGenerationService
                                            .GenerateEmbeddingsAsync([question], this.kernel)
                                            .ConfigureAwait(false);

        var generatedQuestions = await GetEvaluations(answer, context, strictness)
                                        .ToArrayAsync()
                                        .ConfigureAwait(false);

        var generatedQuestionsEmbeddings = await this.textEmbeddingGenerationService
                                            .GenerateEmbeddingsAsync(generatedQuestions.Select(c => c.Question).ToArray(), this.kernel)
                                            .ConfigureAwait(false);

        var evaluations = generatedQuestionsEmbeddings
                        .Select(c => TensorPrimitives.CosineSimilarity(questionEmbeddings.Single().Span, c.Span)
                        *
                        generatedQuestions[generatedQuestionsEmbeddings.IndexOf(c)].Committal)
                        .ToArray();

        return (evaluations.Average(), generatedQuestions.Select((c, index) => (c, evaluations[index])));
    }

    public async Task<(float Score, IEnumerable<(RelevanceEvaluation, float)> Evaluations)> EvaluateAsync(MemoryAnswer answer, int strictness = 3)
    {
        return await EvaluateAsync(
                answer.Question, 
                answer.Result,
                String.Join(Environment.NewLine, answer.RelevantSources.SelectMany(c => c.Partitions.Select(p => p.Text))), 
                strictness)
            .ConfigureAwait(false);
    }

    private async IAsyncEnumerable<RelevanceEvaluation> GetEvaluations(string answer, string context, int strictness)
    {
        foreach (var item in Enumerable.Range(0, strictness))
        {
            var statements = await Try(3, async (remainingTry) =>
            {
                var extraction = await ExtractQuestion.InvokeAsync(this.kernel, new KernelArguments
                {
                    { "context", context },
                    { "answer", answer }
                }).ConfigureAwait(false);

                return JsonSerializer.Deserialize<RelevanceEvaluation>(extraction.GetValue<string>()!);
            }).ConfigureAwait(false);

            yield return statements!;
        }
    }
}
public class RelevanceEvaluation
{
    public string Question { get; set; } = null!;

    public int Committal { get; set; }
}

#pragma warning restore SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Restore this diagnostic to proceed.
#pragma warning restore SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Restore this diagnostic to proceed.
