using UnityEngine;
using System;
using System.Collections.Generic;
using Cysharp.Threading.Tasks;
using LLama;
using LLama.Common;
using LLama.Native;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using static LLama.StatefulExecutorBase;

public class LLamaModelRuntime : MonoBehaviour
{
    public string ModelPath = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
    public int GpuLayerCount = 20;
    public int ContextSize = 2048;
    public bool DisableKqvOffload = true;

    private LLamaWeights _model;
    private LLamaContext _context;
    private InteractiveExecutor _executor;
    private ExecutorBaseState _emptyState;
    private readonly List<ExecutorBaseState> _executorStates = new List<ExecutorBaseState>();
    private readonly List<ChatSession> _chatSessions = new List<ChatSession>();
    private int _activeSession;

    public bool IsInitialized => _executor != null;

    public async UniTask InitializeAsync(int sessionCount, string systemPrompt, CancellationToken cancel = default)
    {
        if (IsInitialized)
        {
            return;
        }

        if (sessionCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sessionCount), "Session count must be > 0.");
        }

        EnsureCudaNativeLibrariesLoaded();
        LogNativeBackendInfo();

        var parameters = new ModelParams(Application.streamingAssetsPath + "/" + ModelPath)
        {
            ContextSize = (uint?)ContextSize,
            Seed = 1337,
            GpuLayerCount = GpuLayerCount,
            NoKqvOffload = DisableKqvOffload
        };

        await UniTask.SwitchToThreadPool();
        cancel.ThrowIfCancellationRequested();
        _model = LLamaWeights.LoadFromFile(parameters);
        _context = _model.CreateContext(parameters);
        _executor = new InteractiveExecutor(_context);
        _emptyState = _executor.GetStateData();

        _chatSessions.Clear();
        _executorStates.Clear();
        for (var i = 0; i < sessionCount; i++)
        {
            var session = new ChatSession(_executor);
            session.AddSystemMessage(systemPrompt);
            _chatSessions.Add(session);
            _executorStates.Add(null);
        }

        _activeSession = 0;
        await UniTask.SwitchToMainThread();
    }

    public void SwitchSession(int index)
    {
        EnsureInitialized();
        if (index < 0 || index >= _chatSessions.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        SaveActiveSession();
        _activeSession = index;
        var state = _executorStates[_activeSession] ?? _emptyState;
        _ = _executor.LoadState(state);
    }

    public IReadOnlyList<ChatHistory.Message> GetActiveMessages()
    {
        EnsureInitialized();
        return _chatSessions[_activeSession].History.Messages;
    }

    public IAsyncEnumerable<string> ChatAsync(string userMessage, InferenceParams inferenceParams)
    {
        EnsureInitialized();
        return _chatSessions[_activeSession].ChatAsync(new ChatHistory.Message(AuthorRole.User, userMessage), inferenceParams);
    }

    public async UniTask<string> CompleteOnceAsync(string systemPrompt, string userPrompt, InferenceParams inferenceParams, CancellationToken cancel = default)
    {
        EnsureInitialized();

        var savedState = _executor.GetStateData();
        try
        {
            var oneShotSession = new ChatSession(_executor);
            if (!string.IsNullOrWhiteSpace(systemPrompt))
            {
                oneShotSession.AddSystemMessage(systemPrompt);
            }

            var parts = new List<string>(128);
            await foreach (var token in ChatConcurrent(oneShotSession.ChatAsync(new ChatHistory.Message(AuthorRole.User, userPrompt), inferenceParams), cancel))
            {
                parts.Add(token);
            }

            return string.Concat(parts).Trim();
        }
        finally
        {
            await _executor.LoadState(savedState);
        }
    }

    private void SaveActiveSession()
    {
        _executorStates[_activeSession] = _executor.GetStateData();
    }

    private static async IAsyncEnumerable<string> ChatConcurrent(IAsyncEnumerable<string> tokens, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancel)
    {
        await UniTask.SwitchToThreadPool();
        await foreach (var token in tokens)
        {
            cancel.ThrowIfCancellationRequested();
            yield return token;
        }
    }

    private void EnsureInitialized()
    {
        if (!IsInitialized)
        {
            throw new InvalidOperationException("LLama runtime is not initialized.");
        }
    }

    private void OnDestroy()
    {
        _context?.Dispose();
        _model?.Dispose();
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