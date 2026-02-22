using UnityEngine;
using System;
using System.IO;
using System.Collections.Generic;
using LLama;
using LLama.Common;
using LLama.Native;
using Cysharp.Threading.Tasks;
using TMPro;
using System.Threading;
using System.Runtime.InteropServices;

public class LLamaCalculator : MonoBehaviour
{
    [Serializable]
    public class FewShotExample
    {
        [TextArea(2, 6)]
        public string UserPrompt;
        [TextArea(2, 6)]
        public string AssistantPrompt;
    }

    public string ModelPath = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
    public int MaxNewTokens = 256;
    public float RepeatPenalty = 1.1f;
    public int RepeatLastTokensCount = 64;
    public int MaxSameTokenInRow = 64;
    public int GpuLayerCount = 20;
    public int ContextSize = 2048;
    public bool DisableKqvOffload = true;
    [TextArea(3, 10)]
    public string SystemPrompt = "You are a precise calculator assistant. Return direct, correct answers.";
    [Tooltip("Optional few-shot examples injected before the first live prompt.")]
    public List<FewShotExample> FewShotExamples = new List<FewShotExample>();

    [Header("UI")]
    public TMP_Text InputText;
    public TMP_Text OutputText;

    private CancellationTokenSource _cts;
    private ChatSession _chatSession;
    private string _inputBuffer = string.Empty;
    private string _pendingPrompt = string.Empty;
    private bool _hasPendingPrompt;

    async UniTaskVoid Start()
    {
        _cts = new CancellationTokenSource();
        try
        {
            EnsureCudaNativeLibrariesLoaded();
            LogNativeBackendInfo();

            RefreshInputText();
            ClearOutput();

            var parameters = new ModelParams(Application.streamingAssetsPath + "/" + ModelPath)
            {
                ContextSize = (uint?)ContextSize,
                Seed = 1337,
                GpuLayerCount = GpuLayerCount,
                NoKqvOffload = DisableKqvOffload
            };

            await UniTask.SwitchToThreadPool();
            using var model = LLamaWeights.LoadFromFile(parameters);
            await UniTask.SwitchToMainThread();

            using var context = model.CreateContext(parameters);
            var executor = new InteractiveExecutor(context);
            _chatSession = new ChatSession(executor);
            _chatSession.AddSystemMessage(SystemPrompt);
            ApplyFewShotExamples(_chatSession);

            await ChatRoutine(_cts.Token);
        }
        catch (OperationCanceledException)
        {
            // Expected during shutdown.
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to initialize LLamaCalculator: {e}");
        }
    }

    public void AppendString(string value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return;
        }

        _inputBuffer += value;
        RefreshInputText();
    }

    public void ClearString()
    {
        _inputBuffer = string.Empty;
        RefreshInputText();
    }

    public void EnterPrompt()
    {
        var trimmedPrompt = _inputBuffer.Trim();
        ClearInputAndOutput();

        if (string.IsNullOrWhiteSpace(trimmedPrompt))
        {
            return;
        }

        _pendingPrompt = trimmedPrompt;
        _hasPendingPrompt = true;
    }

    public void SetFewShotExamples(IEnumerable<FewShotExample> examples)
    {
        FewShotExamples = examples == null ? new List<FewShotExample>() : new List<FewShotExample>(examples);
    }

    public void AddFewShotExample(string userPrompt, string assistantPrompt)
    {
        if (FewShotExamples == null)
        {
            FewShotExamples = new List<FewShotExample>();
        }

        FewShotExamples.Add(new FewShotExample
        {
            UserPrompt = userPrompt ?? string.Empty,
            AssistantPrompt = assistantPrompt ?? string.Empty
        });
    }

    private async UniTask ChatRoutine(CancellationToken cancel = default)
    {
        while (!cancel.IsCancellationRequested)
        {
            await UniTask.WaitUntil(() => _hasPendingPrompt || cancel.IsCancellationRequested);
            if (cancel.IsCancellationRequested)
            {
                break;
            }

            var userMessage = _pendingPrompt;
            _pendingPrompt = string.Empty;
            _hasPendingPrompt = false;

            AppendOutput($"User: {userMessage}\nAssistant: ");

            var inferenceParams = new InferenceParams
            {
                Temperature = 0.6f,
                MaxTokens = MaxNewTokens,
                RepeatPenalty = RepeatPenalty,
                RepeatLastTokensCount = RepeatLastTokensCount,
                AntiPrompts = new List<string> { "<|im_end|>", "\\nUser:", "User:" }
            };

            var previousToken = string.Empty;
            var sameTokenCount = 0;

            await foreach (var token in ChatConcurrent(
                _chatSession.ChatAsync(
                    new ChatHistory.Message(AuthorRole.User, userMessage),
                    inferenceParams,
                    cancel
                )
            ))
            {
                AppendOutput(token);

                if (token == previousToken)
                {
                    sameTokenCount++;
                    if (sameTokenCount >= MaxSameTokenInRow)
                    {
                        Debug.LogWarning($"Stopping generation early: token '{token}' repeated {sameTokenCount} times.");
                        break;
                    }
                }
                else
                {
                    previousToken = token;
                    sameTokenCount = 1;
                }

                await UniTask.NextFrame(cancellationToken: cancel);
            }

            AppendOutput("\n");
        }
    }

    private void RefreshInputText()
    {
        if (InputText != null)
        {
            InputText.text = _inputBuffer;
        }
    }

    private void ClearInputAndOutput()
    {
        _inputBuffer = string.Empty;
        RefreshInputText();
        ClearOutput();
    }

    private void ClearOutput()
    {
        if (OutputText != null)
        {
            OutputText.text = string.Empty;
        }
    }

    private void AppendOutput(string value)
    {
        if (OutputText != null)
        {
            OutputText.text += value;
        }
    }

    private void ApplyFewShotExamples(ChatSession session)
    {
        if (FewShotExamples == null || FewShotExamples.Count == 0)
        {
            return;
        }

        for (var i = 0; i < FewShotExamples.Count; i++)
        {
            var example = FewShotExamples[i];
            if (example == null)
            {
                continue;
            }

            var hasUserPrompt = !string.IsNullOrWhiteSpace(example.UserPrompt);
            var hasAssistantPrompt = !string.IsNullOrWhiteSpace(example.AssistantPrompt);

            if (!hasUserPrompt && !hasAssistantPrompt)
            {
                continue;
            }

            if (hasUserPrompt)
            {
                session.AddUserMessage(example.UserPrompt.Trim());
            }

            if (hasAssistantPrompt)
            {
                session.AddAssistantMessage(example.AssistantPrompt.Trim());
            }

            if (hasUserPrompt != hasAssistantPrompt)
            {
                Debug.LogWarning($"Few-shot example #{i} has only one side of the exchange. Consider providing both prompts.");
            }
        }
    }

    private void OnDestroy()
    {
        _cts?.Cancel();
    }

    private async IAsyncEnumerable<string> ChatConcurrent(IAsyncEnumerable<string> tokens)
    {
        await UniTask.SwitchToThreadPool();
        await foreach (var token in tokens)
        {
            yield return token;
        }
    }

    [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr LoadLibrary(string lpFileName);

    private static void EnsureCudaNativeLibrariesLoaded()
    {
        var searchPaths = new[]
        {
            Path.Combine(Application.dataPath, "Plugins", "x86_64"),
            Application.dataPath
        };

        TryLoadOptional("cudart64_12.dll", searchPaths);
        TryLoadOptional("cublas64_12.dll", searchPaths);
        TryLoadOptional("cublasLt64_12.dll", searchPaths);

        if (!TryLoadRequired("llama.dll", searchPaths) && !TryLoadRequired("libllama.dll", searchPaths))
        {
            throw new DllNotFoundException("Neither llama.dll nor libllama.dll was found in the expected Unity paths.");
        }
    }

    private static bool TryLoadRequired(string fileName, string[] searchPaths)
    {
        foreach (var path in searchPaths)
        {
            var fullPath = Path.Combine(path, fileName);
            if (!File.Exists(fullPath))
            {
                continue;
            }

            var handle = LoadLibrary(fullPath);
            if (handle != IntPtr.Zero)
            {
                Debug.Log($"Loaded native library: {fullPath}");
                return true;
            }

            var error = Marshal.GetLastWin32Error();
            throw new DllNotFoundException($"Failed to load native library '{fullPath}' (Win32 error {error}).");
        }

        return false;
    }

    private static void TryLoadOptional(string fileName, string[] searchPaths)
    {
        foreach (var path in searchPaths)
        {
            var fullPath = Path.Combine(path, fileName);
            if (!File.Exists(fullPath))
            {
                continue;
            }

            var handle = LoadLibrary(fullPath);
            if (handle != IntPtr.Zero)
            {
                Debug.Log($"Loaded CUDA dependency: {fullPath}");
                return;
            }

            var error = Marshal.GetLastWin32Error();
            Debug.LogWarning($"Could not load optional CUDA dependency '{fullPath}' (Win32 error {error}).");
            return;
        }
    }

    private static void LogNativeBackendInfo()
    {
        try
        {
            var systemInfoPtr = NativeApi.llama_print_system_info();
            var systemInfo = systemInfoPtr == IntPtr.Zero ? "<null>" : Marshal.PtrToStringAnsi(systemInfoPtr);
            Debug.Log($"LLama backend loaded. GPU offload: {NativeApi.llama_supports_gpu_offload()}, max devices: {NativeApi.llama_max_devices()}");
            Debug.Log($"LLama system info: {systemInfo}");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"Could not query LLama native backend info: {e.Message}");
        }
    }
}
