using KernelMemory.Evaluation.Evaluators;
using KernelMemory.Evaluation.TestSet;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.KernelMemory;
using Microsoft.KernelMemory.MemoryStorage;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using System.Text.Json;

namespace KernelMemory.Evaluation;

public class TestSetGenerator : EvaluationEngine
{
    public struct Distribution
    {
        public float Simple = .5f;

        public float Reasoning = .16f;

        public float MultiContext = .17f;

        public float Conditioning = .17f;

        public Distribution() { }
    }

    private readonly ServiceProvider serviceProvider;

    private readonly IMemoryDb memory;

    private readonly Kernel evaluatorKernel;

    private Kernel translatorKernel;

    public event EventHandler<TestSetGeneratorProgressionEventArgs> OnTestSetGeneratorProgression;

    private readonly TestSetGeneratorProgressionEventArgs progression;
#pragma warning disable SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

    private KernelFunction Translate => this.translatorKernel.CreateFunctionFromPrompt(this.GetSKPrompt("Transmutation", "Translate"), new OpenAIPromptExecutionSettings
    {
        Temperature = 1e-8f,
        Seed = 0
    });

    private KernelFunction QuestionAnswerGeneration => this.evaluatorKernel.CreateFunctionFromPrompt(this.GetSKPrompt("SyntheticData", "QuestionAnswer"), new OpenAIPromptExecutionSettings
    {
        Temperature = 1e-8f,
        Seed = 0,
        ResponseFormat = "json_object"
    });

    private KernelFunction KeyPhraseExtractionPrompt => this.evaluatorKernel.CreateFunctionFromPrompt(this.GetSKPrompt("Extraction", "Keyphrase"), new OpenAIPromptExecutionSettings
    {
        Temperature = 1e-8f,
        Seed = 0,
        ResponseFormat = "json_object"
    });

    private KernelFunction SeedQuestionGeneration => this.evaluatorKernel.CreateFunctionFromPrompt(this.GetSKPrompt("SyntheticData", "SeedQuestion"), new OpenAIPromptExecutionSettings
    {
        Temperature = 1e-8f,
        Seed = 0
    });

    private KernelFunction ReasoningQuestionGeneration => this.evaluatorKernel.CreateFunctionFromPrompt(this.GetSKPrompt("SyntheticData", "ReasoningQuestion"), new OpenAIPromptExecutionSettings
    {
        Temperature = 1e-8f,
        Seed = 0
    });

    private KernelFunction MultiContextQuestionGeneratoin => this.evaluatorKernel.CreateFunctionFromPrompt(this.GetSKPrompt("SyntheticData", "MultiContextQuestion"), new OpenAIPromptExecutionSettings
    {
        Temperature = 1e-8f,
        Seed = 0,
        ResponseFormat = "json_object"
    });

    private KernelFunction ConditiningQuestionGeneration => this.evaluatorKernel.CreateFunctionFromPrompt(this.GetSKPrompt("SyntheticData", "ConditionalQuestion"), new OpenAIPromptExecutionSettings
    {
        Temperature = 1e-8f,
        Seed = 0,
        ResponseFormat = "json_object"
    });

#pragma warning restore SKEXP0010 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

    public TestSetGenerator(Kernel kernel, IKernelMemoryBuilder memoryBuilder)
    {
        this.serviceProvider = memoryBuilder.Services.BuildServiceProvider();
        this.memory = this.serviceProvider.GetRequiredService<IMemoryDb>();

        this.evaluatorKernel = kernel.Clone();

        this.progression = new TestSetGeneratorProgressionEventArgs
        {
            Total = 0,
            Current = 0
        };
    }

    public async IAsyncEnumerable<TestSet.TestSet> GenerateTestSetsAsync(
        string index,
        int count = 10,
        int retryCount = 3,
        string language = null!,
        Kernel translatorKernel = null!,
        Distribution? distribution = null)
    {
        distribution = distribution ?? new Distribution();

        if (distribution.Value.Simple + distribution.Value.Reasoning + distribution.Value.MultiContext + distribution.Value.Conditioning != 1)
            throw new ArgumentException("The sum of distribution values must be 1.");

        if (!string.IsNullOrEmpty(language) && translatorKernel == null)
            throw new ArgumentNullException("When using translation, you must provide the translation kernel.");

        this.translatorKernel = translatorKernel;

        var simpleCount = (int)(Math.Ceiling(count * distribution.Value.Simple));
        var reasoningCount = (int)(Math.Floor(count * distribution.Value.Reasoning));
        var multiContextCount = (int)(Math.Round(count * distribution.Value.MultiContext));
        var conditioningCount = (int)(Math.Round(count * distribution.Value.Conditioning));

        var documentIds = new List<string>();

        await foreach (var record in this.memory.GetListAsync(index, limit: Int32.MaxValue))
        {
            if (documentIds.Contains(record.GetDocumentId()))
                continue;

            documentIds.Add(record.GetDocumentId());
        }

        this.progression.Total = documentIds.Count * count;
        this.progression.Current = 0;

        if (this.OnTestSetGeneratorProgression != null)
        {
            this.OnTestSetGeneratorProgression(this, this.progression);
        }

        foreach (var documentId in documentIds)
        {
            var partitionRecords = await this.memory.GetListAsync(index,
                        filters: new[] { new MemoryFilter().ByDocument(documentId) },
                        limit: Int32.MaxValue)
                        .ToArrayAsync()
                        .ConfigureAwait(false);

            var nodes = SplitRecordsIntoNodes(partitionRecords, count);

            var questions = GetSimpleQuestionTestSetsAsync(nodes.Take(simpleCount), language: language, retryCount: retryCount)
                                .Concat(GetReasoningTestSetsAsync(nodes.Skip(simpleCount).Take(reasoningCount), language: language, retryCount: retryCount))
                                .Concat(GetMultiContextTestSetsAsync(nodes.Skip(simpleCount + reasoningCount).Take(multiContextCount), language: language, retryCount: retryCount))
                                .Concat(GetConditioningTestSetsAsync(nodes.Skip(simpleCount + reasoningCount + multiContextCount).Take(conditioningCount), language: language, retryCount: retryCount));

            await foreach (var item in questions)
            {
                yield return item;

                this.progression.Current += 1;

                if (OnTestSetGeneratorProgression != null)
                {
                    OnTestSetGeneratorProgression(this, this.progression);
                }
            }
        }
    }

    private async IAsyncEnumerable<TestSet.TestSet> GetMultiContextTestSetsAsync(
        IEnumerable<MemoryRecord[]> nodes,
        string language = null!,
        int retryCount = 3)
    {
        foreach (var partition in nodes)
        {
            if (partition.Length < 2)
                continue;

            var seedQuestionContext = partition.First().GetPartitionText();
            var alternativeContext = partition.Last().GetPartitionText();

            var seedQuestion = await GetQuestionSeedAsync(seedQuestionContext, language, retryCount).ConfigureAwait(false);

            var question = await GetMultiContextQuestionAsync(seedQuestionContext, alternativeContext, seedQuestion, language, retryCount).ConfigureAwait(false);

            var groundTruth = await GetQuestionAnswerAsync(seedQuestionContext + " " + alternativeContext, question, language, retryCount).ConfigureAwait(false);

            var testSet = new TestSet.TestSet
            {
                Question = seedQuestion,
                QuestionType = QuestionType.MultiContext,
                GroundTruth = groundTruth.Answer,
                GroundTruthVerdict = groundTruth.Verdict,
                Context = [seedQuestionContext, alternativeContext]
            };

            yield return testSet;
        }
    }

    private Task<string> GetMultiContextQuestionAsync(string context1, string context2, string seedQuestion, string language = null!, int retryCount = 3)
    {
        return Try(retryCount, async (remainingTry) =>
        {
            var question = await MultiContextQuestionGeneratoin.InvokeAsync(this.evaluatorKernel, new KernelArguments
                {
                    { "question", seedQuestion },
                    { "context1", context1 },
                    { "context2", context2 }
                }).ConfigureAwait(false);

            if (!string.IsNullOrEmpty(language))
            {
                question = await Translate.InvokeAsync(this.translatorKernel, new KernelArguments
                    {
                        { "input", question.GetValue<string>() },
                        { "translate_to", language }
                    }).ConfigureAwait(false);
            }

            return question.GetValue<string>();
        })!;
    }


    private async IAsyncEnumerable<TestSet.TestSet> GetReasoningTestSetsAsync(
        IEnumerable<MemoryRecord[]> nodes,
        string language = null!,
        int retryCount = 3)
    {
        foreach (var partition in nodes)
        {
            var nodeText = string.Join(" ", partition.Select(r => r.GetPartitionText()));

            var seedQuestion = await GetQuestionSeedAsync(nodeText, language, retryCount).ConfigureAwait(false);

            var question = await GetReasoningQuestionAsync(nodeText, seedQuestion, language, retryCount).ConfigureAwait(false);

            var groundTruth = await GetQuestionAnswerAsync(nodeText, question, language, retryCount).ConfigureAwait(false);

            var testSet = new TestSet.TestSet
            {
                Question = seedQuestion,
                QuestionType = QuestionType.Reasoning,
                GroundTruth = groundTruth.Answer,
                GroundTruthVerdict = groundTruth.Verdict,
                Context = partition.Select(r => r.GetPartitionText())
            };

            yield return testSet;
        }
    }

    private Task<string> GetReasoningQuestionAsync(string context, string seedQuestion, string language = null!, int retryCount = 3)
    {
        return Try(retryCount, async (remainingTry) =>
        {
            var question = await ReasoningQuestionGeneration.InvokeAsync(this.evaluatorKernel, new KernelArguments
                {
                    { "question", seedQuestion },
                    { "context", context }
                }).ConfigureAwait(false);

            if (!string.IsNullOrEmpty(language))
            {
                question = await Translate.InvokeAsync(this.translatorKernel, new KernelArguments
                    {
                        { "input", question.GetValue<string>() },
                        { "translate_to", language }
                    }).ConfigureAwait(false);
            }

            return question.GetValue<string>();
        })!;
    }

    private async IAsyncEnumerable<TestSet.TestSet> GetConditioningTestSetsAsync(
        IEnumerable<MemoryRecord[]> nodes,
        string language = null!,
        int retryCount = 3)
    {
        foreach (var partition in nodes)
        {
            var nodeText = string.Join(" ", partition.Select(r => r.GetPartitionText()));

            var seedQuestion = await GetQuestionSeedAsync(nodeText, language, retryCount).ConfigureAwait(false);

            var question = await GetConditionningQuestionAsync(nodeText, seedQuestion, language, retryCount).ConfigureAwait(false);

            var groundTruth = await GetQuestionAnswerAsync(nodeText, question, language, retryCount).ConfigureAwait(false);

            var testSet = new TestSet.TestSet
            {
                Question = seedQuestion,
                QuestionType = QuestionType.Conditioning,
                GroundTruth = groundTruth.Answer,
                GroundTruthVerdict = groundTruth.Verdict,
                Context = partition.Select(r => r.GetPartitionText())
            };

            yield return testSet;
        }
    }

    private Task<string> GetConditionningQuestionAsync(string context, string seedQuestion, string language = null!, int retryCount = 3)
    {
        return Try(retryCount, async (remainingTry) =>
        {
            var question = await ConditiningQuestionGeneration.InvokeAsync(this.evaluatorKernel, new KernelArguments
                {
                    { "question", seedQuestion },
                    { "context", context }
                }).ConfigureAwait(false);

            if (!string.IsNullOrEmpty(language))
            {
                question = await Translate.InvokeAsync(this.translatorKernel, new KernelArguments
                    {
                        { "input", question.GetValue<string>() },
                        { "translate_to", language }
                    }).ConfigureAwait(false);
            }

            return question.GetValue<string>();
        })!;
    }

    private async IAsyncEnumerable<TestSet.TestSet> GetSimpleQuestionTestSetsAsync(
        IEnumerable<MemoryRecord[]> nodes,
        string language = null!,
        int retryCount = 3)
    {
        foreach (var partition in nodes)
        {
            var nodeText = string.Join(" ", partition.Select(r => r.GetPartitionText()));

            var seedQuestion = await GetQuestionSeedAsync(nodeText, language, retryCount).ConfigureAwait(false);

            var groundTruth = await GetQuestionAnswerAsync(nodeText, seedQuestion, language, retryCount).ConfigureAwait(false);

            var testSet = new TestSet.TestSet
            {
                Question = seedQuestion,
                QuestionType = QuestionType.Simple,
                GroundTruth = groundTruth.Answer,
                GroundTruthVerdict = groundTruth.Verdict,
                Context = partition.Select(r => r.GetPartitionText())
            };

            yield return testSet;
        }
    }

    private Task<string> GetQuestionSeedAsync(
            string context,
            string language = null!,
            int retryCount = 3)
    {
        return Try(retryCount, async (remainingTry) =>
        {
            var phrases = await GetKeyPhrases(context, retryCount).ConfigureAwait(false);

            var seedQuestion = await SeedQuestionGeneration.InvokeAsync(this.evaluatorKernel, new KernelArguments
                    {
                        { "keyPhrase", phrases.First() },
                        { "context", context }
                    }).ConfigureAwait(false);

            if (!string.IsNullOrEmpty(language))
            {
                seedQuestion = await Translate.InvokeAsync(this.translatorKernel, new KernelArguments
                    {
                        { "input", seedQuestion.GetValue<string>() },
                        { "translate_to", language }
                    }).ConfigureAwait(false);
            }

            return seedQuestion.GetValue<string>();
        })!;
    }

    private Task<IEnumerable<string>> GetKeyPhrases(string context, int retryCount = 3)
    {
        Random rand = new Random();

        return Try(retryCount, async (remainingTry) =>
        {
            var generatedKeyPhrases = await KeyPhraseExtractionPrompt.InvokeAsync(this.evaluatorKernel, new KernelArguments
                {
                    { "input", context }
            }).ConfigureAwait(false);

            var extraction = JsonSerializer.Deserialize<KeyphraseExtraction>(generatedKeyPhrases.GetValue<string>()!);

            var phrases = extraction!.Keyphrases.ToArray();
            rand.Shuffle(phrases);

            return phrases.AsEnumerable();
        });
    }

    private Task<QuestionAnswer> GetQuestionAnswerAsync(string context, string question, string language = null!, int retryCount = 3)
    {
        return Try(retryCount, async (remainingTry) =>
        {
            var generatedAnswer = await QuestionAnswerGeneration.InvokeAsync(this.evaluatorKernel, new KernelArguments
                {
                    { "context", context },
                    { "question", question }
                }).ConfigureAwait(false);

            var answer = JsonSerializer.Deserialize<QuestionAnswer>(generatedAnswer.GetValue<string>()!);

            if (answer!.Verdict <= 0 && remainingTry > 0)
            {
                throw new InvalidDataException();
            }

            if (!string.IsNullOrEmpty(language))
            {
                generatedAnswer = await Translate.InvokeAsync(this.translatorKernel, new KernelArguments
                    {
                        { "input", answer.Answer },
                        { "translate_to", language }
                    }).ConfigureAwait(false);

                answer.Answer = generatedAnswer.GetValue<string>()!;
            }

            return answer;
        });
    }

    /// <summary>
    /// Split records into nodes
    /// </summary>
    /// <param name="records">The records to create nodes.</param>
    /// <param name="count">The nomber of nodes to create.</param>
    /// <returns></returns>
    private IEnumerable<MemoryRecord[]> SplitRecordsIntoNodes(IEnumerable<MemoryRecord> records, int count)
    {
        var groups = new List<MemoryRecord[]>();
        var groupSize = (int)Math.Round((double)records.Count() / count);

        for (int i = 0; i < count; i++)
        {
            var group = records
                .Skip(i * (groupSize > 0 || records.Count() < count ? groupSize : 0))
                .Take(groupSize > 0 ? groupSize : records.Count())
                .ToArray();

            groups.Add(group);
        }

        return groups;
    }
}
