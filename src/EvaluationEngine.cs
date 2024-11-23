using System.Reflection;

namespace KernelMemory.Evaluation;

public abstract class EvaluationEngine
{
    protected string GetSKPrompt(string pluginName, string functionName)
    {
        var resourceStream = Assembly.GetExecutingAssembly()
                                     .GetManifestResourceStream($"Plugins/{pluginName}/{functionName}.txt");

        using var reader = new StreamReader(resourceStream!);
        var text = reader.ReadToEnd();
        return text;
    }

    protected async Task<T> Try<T>(int maxCount, Func<int, Task<T>> action)
    {
        do
        {
            try
            {
                return await action(maxCount).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                if (maxCount == 0)
                {
                    throw;
                }
            }
        } while (maxCount-- > 0);

        throw new InvalidProgramException();
    }
}
