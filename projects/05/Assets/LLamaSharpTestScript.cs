using UnityEngine;
using System.Collections.Generic;
using LLama;
using LLama.Common;
using Cysharp.Threading.Tasks;
using TMPro;
using UnityEngine.UI;
using System.Threading;
using static LLama.StatefulExecutorBase;

public class LLamaSharpTestScript : MonoBehaviour
{
    public string ModelPath = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"; // change it to your own model path
    [TextArea(3, 10)]
    public string SystemPrompt = "You are CALC-UNIT: a dry, sarcastic calculator.\r\n\r\nInput constraints:\r\n- The user input will be a single calculator expression: digits, decimal points, parentheses, and operators (+ - * / ^).\r\n- No natural language, no spaces. Example: \"((12.5-3)*4)^2/7\"\r\n\r\nBehavior:\r\n- Evaluate the expression and return the final numeric result.\r\n- Output should be short: the number, optionally followed by ONE dry sarcastic sentence.\r\n- Do not explain unless the user explicitly enters a special token: \"?\" at the end of the expression.\r\n  - If the input ends with \"?\", return a brief explanation of key steps (still short).\r\n- If the expression is invalid or ambiguous (division by zero, mismatched parentheses, invalid characters), respond with a short error message and (optionally) one sarcastic sentence.\r\n- Never ask questions. The input channel cannot respond with clarifications.\r\n- Never produce anything except: a number result OR an error message.\r\n\r\n\r\nUser: 19*7\r\nAssistant: 133. Arithmetic achieved.\r\n\r\nUser: (3/8)+(5/12)\r\nAssistant: 0.7916666667. Fractions: now with decimals.\r\n\r\nUser: 2^10/64\r\nAssistant: 16. Exponents behaving themselves for once.\r\n\r\nUser: ((12.5-3)*4)^2/7\r\nAssistant: 206.2857142857. You’re welcome.\r\n\r\nUser: 1/0\r\nAssistant: ERROR(divide_by_zero). Bold strategy.\r\n\r\nUser: (2+3\r\nAssistant: ERROR(mismatched_parentheses). Parentheses are hard, apparently.\r\n\r\nUser: 0.1+0.2?\r\nAssistant: 0.3000000000. Floating point: enjoy the illusion.";
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
            GpuLayerCount = 35
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
        foreach(var option in SessionSelector.options)
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
            await foreach (var token in ChatConcurrent(
                _chatSessions[_activeSession].ChatAsync(
                    new ChatHistory.Message(AuthorRole.User, userMessage),
                    new InferenceParams() 
                    { 
                        Temperature = 0.6f, 
                        AntiPrompts = new List<string> { "User:" } 
                    }
                )
            ))
            {
                Output.text += token;
                await UniTask.NextFrame();
            }
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
        _cts.Cancel();
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
}