
using KernelMemory.Evaluation.Evaluators;

namespace KernelMemory.Evaluation;

public class EvaluationMetrics
{
    /// <summary>
    /// Scores the semantic similarity of ground truth with generated answer.
    /// </summary>
    public float AnswerSemanticSimilarity { get; set; }

    /// <summary>
    /// Scores the relevancy of the answer according to the given question.
    /// </summary>
    public (float Score, IEnumerable<(RelevanceEvaluation, float)> Evaluations) AnswerRelevancy { get; set; }

    /// <summary>
    /// Measures answer correctness compared to ground truth as a combination of factuality and semantic similarity.
    /// </summary>
    public (float Score, CorrectnessEvaluation? Evaluation) AnswerCorrectness { get; set; }

    /// <summary>
    /// Measures the factual consistency of the generated answer against the given context.
    /// </summary>
    public (float Score, IEnumerable<StatementEvaluation>? Evaluations) Faithfulness { get; set; }

    /// <summary>
    /// Average Precision is a metric that evaluates whether all of the relevant items selected by the model are ranked higher or not.
    /// </summary>
    public (float Score, IEnumerable<ContextRelevancy>? Evaluations) ContextPrecision { get; set; }

    /// <summary>
    /// Estimates context recall by estimating TP and FN using annotated answer and retrieved context.
    /// </summary>
    public (float Score, IEnumerable<GroundTruthClassification>? Evaluations) ContextRecall { get; set; }
}
