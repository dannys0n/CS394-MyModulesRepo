using UnityEngine;
using System;
using System.IO;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using LLama;
using LLama.Common;
using LLama.Grammars;
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
    public int GpuLayerCount = 20;
    public int ContextSize = 2048;
    public bool DisableKqvOffload = true;
    [TextArea(3, 10)]
    public string SystemPrompt = "You are Bob, a helpful, kind, honest, and precise assistant.";
    public TMP_Text Output;
    public TMP_InputField Input;
    public TMP_Dropdown SessionSelector;
    public Button Submit;
    public enum NpcBehavior
    {
        Guard,
        Cautious,
        Aggressive,
        Scout
    }

    [Serializable]
    public struct NpcDecisionRequest
    {
        public Vector2Int NpcGrid;
        public Vector2Int PlayerPingGrid;
        public NpcBehavior Behavior;
        public int GridWidth;
        public int GridHeight;
        public int NearRadius;
    }

    [Serializable]
    public struct NpcDecision
    {
        public string action;
        public int target_x;
        public int target_y;
    }

    private const string NpcDecisionGrammarGbnf =
        "root ::= ws \"{\" ws \"\\\"action\\\"\" ws \":\" ws action ws \",\" ws \"\\\"target_x\\\"\" ws \":\" ws int ws \",\" ws \"\\\"target_y\\\"\" ws \":\" ws int ws \"}\" ws\n" +
        "action ::= \"\\\"hold\\\"\" | \"\\\"move_to_ping\\\"\" | \"\\\"move_near_self\\\"\"\n" +
        "int ::= \"-\"? [0-9] [0-9]*\n" +
        "ws ::= [ \\t\\n\\r]*\n";

    private static readonly Grammar NpcDecisionGrammar = Grammar.Parse(NpcDecisionGrammarGbnf, "root");
    private static readonly JsonSerializerOptions NpcDecisionJsonOptions = new JsonSerializerOptions { IncludeFields = true };

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
                ContextSize = (uint?)this.ContextSize,
                Seed = 1337,
                GpuLayerCount = this.GpuLayerCount,
                NoKqvOffload = this.DisableKqvOffload
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



    public async UniTask<NpcDecision> DecideNpcGridAsync(NpcDecisionRequest request, CancellationToken cancel = default)
    {
        if (_chatSessions.Count == 0)
        {
            throw new InvalidOperationException("LLama runtime is not initialized yet.");
        }

        request = NormalizeNpcDecisionRequest(request);

        var executor = _chatSessions[_activeSession].Executor as InteractiveExecutor;
        if (executor == null)
        {
            throw new InvalidOperationException("Active executor is not InteractiveExecutor.");
        }

        var savedState = executor.GetStateData();
        try
        {
            var planningSession = new ChatSession(executor);
            planningSession.AddSystemMessage("You are an NPC tactical planner for a grid game. Respond with exactly one JSON object.");

            using var grammarHandle = NpcDecisionGrammar.CreateInstance();
            var inferenceParams = new InferenceParams()
            {
                Temperature = 0.15f,
                MaxTokens = 64,
                RepeatPenalty = 1.05f,
                RepeatLastTokensCount = 32,
                Grammar = grammarHandle
            };

            var prompt = BuildNpcDecisionPrompt(request);
            var json = await CollectResponseAsync(
                planningSession.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), inferenceParams),
                cancel);

            if (!TryParseNpcDecision(json, request, out var decision))
            {
                Debug.LogWarning($"NPC grammar parse failed. Raw output: {json}");
                return BuildFallbackNpcDecision(request);
            }

            return decision;
        }
        finally
        {
            await executor.LoadState(savedState);
        }
    }

    private static NpcDecisionRequest NormalizeNpcDecisionRequest(NpcDecisionRequest request)
    {
        request.GridWidth = Mathf.Max(1, request.GridWidth);
        request.GridHeight = Mathf.Max(1, request.GridHeight);
        request.NearRadius = Mathf.Max(1, request.NearRadius);
        request.NpcGrid = ClampToGrid(request.NpcGrid, request.GridWidth, request.GridHeight);
        request.PlayerPingGrid = ClampToGrid(request.PlayerPingGrid, request.GridWidth, request.GridHeight);
        return request;
    }

    private static Vector2Int ClampToGrid(Vector2Int point, int gridWidth, int gridHeight)
    {
        return new Vector2Int(
            Mathf.Clamp(point.x, 0, gridWidth - 1),
            Mathf.Clamp(point.y, 0, gridHeight - 1));
    }

    private static string BuildNpcDecisionPrompt(NpcDecisionRequest request)
    {
        var behavior = request.Behavior.ToString().ToLowerInvariant();
        return
            "Pick one NPC action from: hold, move_to_ping, move_near_self.\n" +
            $"Grid width={request.GridWidth}, height={request.GridHeight}.\n" +
            $"NPC at x={request.NpcGrid.x}, y={request.NpcGrid.y}.\n" +
            $"Player ping at x={request.PlayerPingGrid.x}, y={request.PlayerPingGrid.y}.\n" +
            $"NPC behavior={behavior}. Near radius={request.NearRadius}.\n" +
            "Rules:\n" +
            "- aggressive prefers move_to_ping.\n" +
            "- cautious prefers move_near_self or hold.\n" +
            "- guard prefers hold unless ping is very close.\n" +
            "- scout prefers move_near_self, but can move_to_ping if beneficial.\n" +
            "- target_x and target_y must be inside the grid.\n" +
            "- if action is hold, target should be NPC position.\n" +
            "- if action is move_near_self, target should stay within near radius of NPC.\n" +
            "Respond with JSON only.";
    }

    private async UniTask<string> CollectResponseAsync(IAsyncEnumerable<string> tokens, CancellationToken cancel)
    {
        var buffer = new StringBuilder(128);
        await foreach (var token in ChatConcurrent(tokens))
        {
            cancel.ThrowIfCancellationRequested();
            buffer.Append(token);
        }

        return buffer.ToString().Trim();
    }

    private static bool TryParseNpcDecision(string json, NpcDecisionRequest request, out NpcDecision decision)
    {
        decision = default;
        if (string.IsNullOrWhiteSpace(json))
        {
            return false;
        }

        try
        {
            decision = JsonSerializer.Deserialize<NpcDecision>(json, NpcDecisionJsonOptions);
        }
        catch
        {
            return false;
        }

        if (string.IsNullOrWhiteSpace(decision.action))
        {
            return false;
        }

        decision.action = decision.action.Trim();
        if (decision.action != "hold" && decision.action != "move_to_ping" && decision.action != "move_near_self")
        {
            return false;
        }

        decision.target_x = Mathf.Clamp(decision.target_x, 0, request.GridWidth - 1);
        decision.target_y = Mathf.Clamp(decision.target_y, 0, request.GridHeight - 1);
        return true;
    }

    private static NpcDecision BuildFallbackNpcDecision(NpcDecisionRequest request)
    {
        return new NpcDecision
        {
            action = "hold",
            target_x = request.NpcGrid.x,
            target_y = request.NpcGrid.y
        };
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








