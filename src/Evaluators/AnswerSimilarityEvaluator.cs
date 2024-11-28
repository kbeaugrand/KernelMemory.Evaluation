using System.Numerics.Tensors;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.KernelMemory;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;

namespace KernelMemory.Evaluation.Evaluators;

#pragma warning disable SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

internal class AnswerSimilarityEvaluator : EvaluationEngine
{
    private readonly Kernel kernel;

    private ITextEmbeddingGenerationService textEmbeddingGenerationService;

    public AnswerSimilarityEvaluator(Kernel kernel)
    {
        this.kernel = kernel.Clone();

        this.textEmbeddingGenerationService = this.kernel.Services.GetRequiredService<ITextEmbeddingGenerationService>();
    }

    public async Task<float> EvaluateAsync(string truth, string answer)
    {
        var answerEmbeddings = await this.textEmbeddingGenerationService
                                            .GenerateEmbeddingsAsync([truth, answer], this.kernel)
                                            .ConfigureAwait(false);

        var evaluation = TensorPrimitives.CosineSimilarity(answerEmbeddings.First().Span, answerEmbeddings.Last().Span);

        return evaluation;
    }

    public async Task<float> EvaluateAsync(TestSet.TestSet testSet, MemoryAnswer answer)
    {
        return await EvaluateAsync(testSet.GroundTruth, answer.Result)
                .ConfigureAwait(false);
    }
}

#pragma warning restore SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Restore this diagnostic to proceed.
