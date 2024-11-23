﻿using System.Diagnostics;
using KernelMemory.Evaluation.Evaluators;
using Microsoft.KernelMemory;
using Microsoft.SemanticKernel;

namespace KernelMemory.Evaluation;

#pragma warning disable SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

public class TestSetEvaluator
{
    private readonly IKernelMemory kernelMemory;
    private readonly string indexName;

    private readonly Kernel evaluatorKernel;

    private FaithfulnessEvaluator Faithfulness => new(this.evaluatorKernel);

    private RelevanceEvaluator Relevance => new(this.evaluatorKernel);

    private AnswerSimilarityEvaluator AnswerSimilarity => new(this.evaluatorKernel);

    private ContextRelevancyEvaluator ContextRelevancy => new(this.evaluatorKernel);

    private AnswerCorrectnessEvaluator AnswerCorrectness => new(this.evaluatorKernel);

    private ContextRecallEvaluator ContextRecall => new(this.evaluatorKernel);

    public TestSetEvaluator(IKernelBuilder evaluatorKernel, IKernelMemory kernelMemory, string indexName)
    {
        this.evaluatorKernel = evaluatorKernel.Build();

        this.kernelMemory = kernelMemory;
        this.indexName = indexName;
    }

    public async IAsyncEnumerable<QuestionEvaluation> EvaluateTestSetAsync(IEnumerable<TestSet.TestSet> questions)
    {
        foreach (var test in questions)
        {
            var stopWatch = Stopwatch.StartNew();

            var answer = await this.kernelMemory
                                    .AskAsync(test.Question, this.indexName)
                                    .ConfigureAwait(false);
            stopWatch.Stop();

            if (answer.NoResult)
            {
                yield return new QuestionEvaluation
                {
                    TestSet = test,
                    MemoryAnswer = answer,
                    Metrics = new(),
                    Elapsed = stopWatch.Elapsed,
                };

                continue;
            }

            var metadata = new Dictionary<string, object?>
            {
                { "Question", test.Question },
                { "IndexName", this.indexName }
            };

            yield return new QuestionEvaluation
            {
                TestSet = test,
                MemoryAnswer = answer,
                Elapsed = stopWatch.Elapsed,
                Metrics = new()
                {
                    AnswerRelevancy = await Relevance.Evaluate(answer, metadata).ConfigureAwait(false),
                    AnswerSemanticSimilarity = await AnswerSimilarity.Evaluate(test, answer, metadata).ConfigureAwait(false),
                    AnswerCorrectness = await AnswerCorrectness.Evaluate(test, answer, metadata).ConfigureAwait(false),
                    Faithfulness = await Faithfulness.Evaluate(answer, metadata).ConfigureAwait(false),
                    ContextPrecision = await ContextRelevancy.Evaluate(answer, metadata).ConfigureAwait(false),
                    ContextRecall = await ContextRecall.Evaluate(test, answer, metadata).ConfigureAwait(false),
                }
            };
        }
    }
}

#pragma warning restore SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

public class QuestionEvaluation
{
    public TestSet.TestSet TestSet { get; set; } = null!;

    public MemoryAnswer MemoryAnswer { get; set; } = null!;

    public TimeSpan Elapsed { get; set; }

    public EvaluationMetrics Metrics { get; set; } = new EvaluationMetrics();

    public Dictionary<string, object?> Metadata { get; set; } = new();
}