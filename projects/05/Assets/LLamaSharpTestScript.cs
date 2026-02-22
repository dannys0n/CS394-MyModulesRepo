using UnityEngine;
using System;
using System.Collections.Generic;
using System.Threading;
using Cysharp.Threading.Tasks;
using LLama.Common;
using TMPro;
using UnityEngine.UI;

public class LLamaSharpTestScript : MonoBehaviour
{
    [Header("Runtime")]
    public LLamaModelRuntime Runtime;
    [TextArea(3, 10)]
    public string SystemPrompt = "You are Bob, a helpful, kind, honest, and precise assistant.";

    [Header("Chat Sampling")]
    public int MaxNewTokens = 256;
    public float RepeatPenalty = 1.1f;
    public int RepeatLastTokensCount = 64;
    public int MaxSameTokenInRow = 64;

    [Header("UI")]
    public TMP_Text Output;
    public TMP_InputField Input;
    public TMP_Dropdown SessionSelector;
    public Button Submit;

    private string _submittedText = string.Empty;
    private CancellationTokenSource _cts;

    private async UniTaskVoid Start()
    {
        _cts = new CancellationTokenSource();
        try
        {
            if (Runtime == null)
            {
                Runtime = GetComponent<LLamaModelRuntime>();
            }

            if (Runtime == null)
            {
                throw new InvalidOperationException("Assign LLamaModelRuntime on this GameObject.");
            }

            SetInteractable(false);
            Submit.onClick.AddListener(OnSubmitClicked);
            SessionSelector.onValueChanged.AddListener(SwitchSession);

            Output.text = "User: ";
            await Runtime.InitializeAsync(SessionSelector.options.Count, SystemPrompt, _cts.Token);
            await ChatRoutine(_cts.Token);
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to initialize LLamaSharp runtime: {e}");
            SetInteractable(false);
        }
        finally
        {
            Submit.onClick.RemoveListener(OnSubmitClicked);
            SessionSelector.onValueChanged.RemoveListener(SwitchSession);
        }
    }

    public async UniTask ChatRoutine(CancellationToken cancel = default)
    {
        while (!cancel.IsCancellationRequested)
        {
            SetInteractable(true);
            await UniTask.WaitUntil(() => !string.IsNullOrEmpty(_submittedText), cancellationToken: cancel);

            var userMessage = _submittedText;
            _submittedText = string.Empty;
            Output.text += " " + userMessage + "\n";
            SetInteractable(false);

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

            await foreach (var token in ChatConcurrent(Runtime.ChatAsync(userMessage, inferenceParams), cancel))
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

                await UniTask.NextFrame(cancel);
            }

            Output.text += "\n";
        }
    }

    private void OnSubmitClicked()
    {
        _submittedText = Input.text;
        Input.text = string.Empty;
    }

    private void SwitchSession(int index)
    {
        Runtime.SwitchSession(index);
        Output.text = "User: ";
        foreach (var message in Runtime.GetActiveMessages())
        {
            if (message.AuthorRole == AuthorRole.System)
            {
                continue;
            }

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

    private static async IAsyncEnumerable<string> ChatConcurrent(IAsyncEnumerable<string> tokens, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancel)
    {
        await UniTask.SwitchToThreadPool();
        await foreach (var token in tokens)
        {
            cancel.ThrowIfCancellationRequested();
            yield return token;
        }
    }

    private void SetInteractable(bool interactable)
    {
        Submit.interactable = interactable;
        Input.interactable = interactable;
        SessionSelector.interactable = interactable;
    }

    private void OnDestroy()
    {
        _cts?.Cancel();
    }
}