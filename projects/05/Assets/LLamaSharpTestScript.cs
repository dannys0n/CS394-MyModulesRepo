using UnityEngine;
using System;
using System.IO;
using System.Collections.Generic;
using LLama;
using LLama.Common;
using LLama.Native;
using Cysharp.Threading.Tasks;
using TMPro;
using UnityEngine.UI;
using System.Threading;
using System.Runtime.InteropServices;
using static LLama.StatefulExecutorBase;

public class LLamaSharpTestScript : MonoBehaviour
{
    public string ModelPath = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"; // change it to your own model path
    public int MaxNewTokens = 256;
    public float RepeatPenalty = 1.1f;
    public int RepeatLastTokensCount = 64;
    public int MaxSameTokenInRow = 64;
    public int GpuLayerCount = 35;
    [TextArea(3, 10)]
    public string SystemPrompt = "You are Bob, a helpful, kind, honest, and precise assistant.";
    public TMP_Text Output;
    public TMP_InputField Input;
    public TMP_Dropdown SessionSelector;
    public Button Submit;

    private ExecutorBaseState _emptyState;
    private List<ExecutorBaseState> _executorStates = new List<ExecutorBaseState>();
    private List<ChatSession> _chatSessions = new List<ChatSession>();
    private int _activeSession = 0;

    private string _submittedText = "";
    private CancellationTokenSource _cts;

    async UniTaskVoid Start()
    {
        _cts = new CancellationTokenSource();
        try
        {
            EnsureCudaNativeLibrariesLoaded();
            LogNativeBackendInfo();

            SetInteractable(false);
            Submit.onClick.AddListener(() =>
            {
                _submittedText = Input.text;
                Input.text = "";
            });
            Output.text = "User: ";
            // Load a model
            var parameters = new ModelParams(Application.streamingAssetsPath + "/" + ModelPath)
            {
                ContextSize = 4096,
                Seed = 1337,
                GpuLayerCount = this.GpuLayerCount
            };
            // Switch to the thread pool for long-running operations
            await UniTask.SwitchToThreadPool();
            using var model = LLamaWeights.LoadFromFile(parameters);
            await UniTask.SwitchToMainThread();
            // Initialize a chat session
            using var context = model.CreateContext(parameters);
            var ex = new InteractiveExecutor(context);
            // Save the empty state for cases when we need to switch to empty session
            _emptyState = ex.GetStateData();
            foreach (var option in SessionSelector.options)
            {
                var session = new ChatSession(ex);
                // This won't process the system prompt until the first user message is received
                // to pre-process it you'd need to look into context.Decode() method.
                // Create an issue on github if you need help with that.
                session.AddSystemMessage(SystemPrompt);
                _chatSessions.Add(session);
                _executorStates.Add(null);
            }
            SessionSelector.onValueChanged.AddListener(SwitchSession);
            _activeSession = 0;
            // run the inference in a loop to chat with LLM
            await ChatRoutine(_cts.Token);
            Submit.onClick.RemoveAllListeners();
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to initialize LLamaSharp CUDA runtime: {e}");
            SetInteractable(false);
        }
    }

    /// <summary>
    /// Chat routine that sends user messages to the chat session and receives responses.
    /// </summary>
    /// <param name="session">Active chat session</param>
    /// <param name="cancel">Cancellation token to stop the routine</param>
    /// <returns></returns>
    public async UniTask ChatRoutine(CancellationToken cancel = default)
    {
        var userMessage = "";
        while (!cancel.IsCancellationRequested)
        {
            // Allow input and wait for the user to submit a message or switch the session
            SetInteractable(true);
            await UniTask.WaitUntil(() => _submittedText != "");
            userMessage = _submittedText;
            _submittedText = "";
            Output.text += " " + userMessage + "\n";
            // Disable input while processing the message
            SetInteractable(false);

            var inferenceParams = new InferenceParams()
            {
                Temperature = 0.6f,
                MaxTokens = MaxNewTokens,
                RepeatPenalty = RepeatPenalty,
                RepeatLastTokensCount = RepeatLastTokensCount,
                AntiPrompts = new List<string> { "<|im_end|>", "\\nUser:", "User:" }
            };

            var previousToken = "";
            var sameTokenCount = 0;

            await foreach (var token in ChatConcurrent(
                _chatSessions[_activeSession].ChatAsync(
                    new ChatHistory.Message(AuthorRole.User, userMessage),
                    inferenceParams
                )
            ))
            {
                Output.text += token;

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

                await UniTask.NextFrame();
            }

            Output.text += "\n";
        }
    }


    private void SwitchSession(int index)
    {
        SaveActiveSession();
        SetActiveSession(index);
    }

    /// <summary>
    /// Saves the state of the active chat session executor.
    /// </summary>
    private void SaveActiveSession()
    {
        _executorStates[_activeSession] = (_chatSessions[_activeSession].Executor as InteractiveExecutor).GetStateData();
    }

    /// <summary>
    /// Sets the active chat session and loads its state.
    /// If the session has a saved state, it loads it. Otherwise, it loads an empty state.
    /// </summary>
    /// <param name="index"></param>
    private void SetActiveSession(int index)
    {
        _activeSession = index;
        if (_executorStates[_activeSession] != null)
        {
            (_chatSessions[_activeSession].Executor as InteractiveExecutor).LoadState(_executorStates[_activeSession]);
        }
        else
        {
            (_chatSessions[_activeSession].Executor as InteractiveExecutor).LoadState(_emptyState);
        }
        Output.text = "User: ";
        foreach (var message in _chatSessions[_activeSession].History.Messages)
        {
            // Skip system prompt
            if (message.AuthorRole != AuthorRole.System)
            {
                // Do not add a new line to the last message
                if (!message.Content.Trim().EndsWith("User:"))
                {
                    Output.text += message.Content + "\n";
                }
                else
                {
                    Output.text += message.Content;
                }
            }
        }
    }

    /// <summary>
    /// Cancels the chat routine when the object is destroyed.
    /// </summary>
    private void OnDestroy()
    {
        _cts?.Cancel();
    }

    /// <summary>
    /// Wraps AsyncEnumerable with transition to the thread pool.
    /// </summary>
    /// <param name="tokens"></param>
    /// <returns>IAsyncEnumerable computed on a thread pool</returns>
    private async IAsyncEnumerable<string> ChatConcurrent(IAsyncEnumerable<string> tokens)
    {
        await UniTask.SwitchToThreadPool();
        await foreach (var token in tokens)
        {
            yield return token;
        }
    }

    /// <summary>
    /// Sets the interactable property of the UI elements.
    /// </summary>
    /// <param name="interactable"></param>
    private void SetInteractable(bool interactable)
    {
        Submit.interactable = interactable;
        Input.interactable = interactable;
        SessionSelector.interactable = interactable;
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

        // Prefer CUDA backend from Plugins/x86_64. Keep libllama as fallback.
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






