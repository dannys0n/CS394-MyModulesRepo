using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Cysharp.Threading.Tasks;
using LLama.Common;
using LLama.Native;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

public class LLamaGridPingController : MonoBehaviour
{
    public LLamaModelRuntime Runtime;
    public LLamaNpcGrammarPlanner Planner;
    public GridLayoutGroup Grid;

    [Header("Runtime Init")]
    public bool AutoInitializeRuntime = true;
    public int RuntimeSessionCount = 1;
    public bool ClearHistoryPerPrompt = true;

    [Header("NPC")]
    public Vector2Int NpcGrid = new Vector2Int(0, 0);
    public LLamaNpcGrammarPlanner.NpcBehavior Behavior = LLamaNpcGrammarPlanner.NpcBehavior.Scout;
    public int NearRadius = 3;
    public TMP_Dropdown BehaviorDropdown;
    public bool PopulateBehaviorDropdownOptions = true;

    [Header("Grid Labels")]
    public string NpcCellLabel = "NPC";
    public string PlayerCellLabel = "player";
    public string OverlapCellLabel = "player + NPC";

    [Header("Output")]
    public TMP_Text SystemPromptOutput;
    public TMP_Text UserPromptOutput;
    public TMP_Text CompletionOutput;
    public TMP_Text DecisionOutput;

    private int _gridWidth;
    private int _gridHeight;
    private readonly List<Button> _gridButtons = new List<Button>();
    private readonly Dictionary<int, TMP_Text> _gridButtonTextByIndex = new Dictionary<int, TMP_Text>();
    private bool _sessionPrimedWithSystemPrompt;
    private string _lastAppliedSystemPrompt = string.Empty;

    private void Awake()
    {
        if (Runtime == null)
        {
            Runtime = GetComponent<LLamaModelRuntime>();
        }

        if (Planner == null)
        {
            Planner = GetComponent<LLamaNpcGrammarPlanner>();
        }

        BindButtons();
        BindBehaviorSelector();
    }

    private async void Start()
    {
        ClearGridButtonTexts();

        if (!AutoInitializeRuntime)
        {
            return;
        }

        await EnsureRuntimeInitializedAsync(this.GetCancellationTokenOnDestroy());
    }

    private void BindButtons()
    {
        _gridButtons.Clear();
        _gridButtonTextByIndex.Clear();

        if (Grid == null)
        {
            Debug.LogError("LLamaGridPingController requires GridLayoutGroup reference.");
            return;
        }

        var buttonCount = Grid.transform.childCount;
        _gridWidth = ResolveGridWidth(buttonCount);
        _gridHeight = Mathf.Max(1, Mathf.CeilToInt(buttonCount / (float)_gridWidth));

        for (var i = 0; i < buttonCount; i++)
        {
            var child = Grid.transform.GetChild(i);
            var button = child.GetComponent<Button>();
            if (button == null)
            {
                continue;
            }

            _gridButtons.Add(button);
            _gridButtonTextByIndex[i] = button.GetComponentInChildren<TMP_Text>(true);

            var index = i;
            button.onClick.RemoveAllListeners();
            button.onClick.AddListener(() => HandleGridClick(index).Forget());
        }
    }

    private void BindBehaviorSelector()
    {
        if (BehaviorDropdown == null)
        {
            return;
        }

        if (PopulateBehaviorDropdownOptions)
        {
            BehaviorDropdown.ClearOptions();
            BehaviorDropdown.AddOptions(new List<string>(Enum.GetNames(typeof(LLamaNpcGrammarPlanner.NpcBehavior))));
        }

        if (BehaviorDropdown.options == null || BehaviorDropdown.options.Count == 0)
        {
            Debug.LogWarning("BehaviorDropdown has no options. Assign options in inspector or enable PopulateBehaviorDropdownOptions.");
            return;
        }

        var currentIndex = Mathf.Clamp((int)Behavior, 0, BehaviorDropdown.options.Count - 1);
        BehaviorDropdown.SetValueWithoutNotify(currentIndex);
        BehaviorDropdown.onValueChanged.RemoveListener(SetBehaviorFromDropdown);
        BehaviorDropdown.onValueChanged.AddListener(SetBehaviorFromDropdown);
        SetBehaviorFromDropdown(currentIndex);
    }

    public void SetBehavior(LLamaNpcGrammarPlanner.NpcBehavior behavior)
    {
        Behavior = behavior;
    }

    public void SetBehaviorFromDropdown(int behaviorIndex)
    {
        if (behaviorIndex < 0 || behaviorIndex >= Enum.GetValues(typeof(LLamaNpcGrammarPlanner.NpcBehavior)).Length)
        {
            Debug.LogWarning($"Invalid behavior index: {behaviorIndex}");
            return;
        }

        Behavior = (LLamaNpcGrammarPlanner.NpcBehavior)behaviorIndex;
    }

    private int ResolveGridWidth(int buttonCount)
    {
        if (Grid.constraint == GridLayoutGroup.Constraint.FixedColumnCount)
        {
            return Mathf.Max(1, Grid.constraintCount);
        }

        if (Grid.constraint == GridLayoutGroup.Constraint.FixedRowCount)
        {
            var rows = Mathf.Max(1, Grid.constraintCount);
            return Mathf.Max(1, Mathf.CeilToInt(buttonCount / (float)rows));
        }

        return Mathf.Max(1, Mathf.RoundToInt(Mathf.Sqrt(buttonCount)));
    }

    private async UniTask<bool> EnsureRuntimeInitializedAsync(CancellationToken cancel)
    {
        if (Planner == null)
        {
            Debug.LogError("LLamaGridPingController requires LLamaNpcGrammarPlanner reference.");
            return false;
        }

        if (Runtime == null)
        {
            Runtime = GetComponent<LLamaModelRuntime>();
        }

        if (Runtime == null)
        {
            Debug.LogError("LLamaGridPingController could not find LLamaModelRuntime.");
            return false;
        }

        if (Runtime.IsInitialized)
        {
            return true;
        }

        SetGridInteractable(false);
        try
        {
            await Runtime.InitializeAsync(Mathf.Max(1, RuntimeSessionCount), string.Empty, cancel);
            _sessionPrimedWithSystemPrompt = false;
            _lastAppliedSystemPrompt = string.Empty;
            return true;
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to initialize LLama runtime for grid planner: {ex}");
            return false;
        }
        finally
        {
            SetGridInteractable(Runtime != null && Runtime.IsInitialized);
        }
    }

    private void SetGridInteractable(bool interactable)
    {
        foreach (var button in _gridButtons)
        {
            if (button != null)
            {
                button.interactable = interactable;
            }
        }
    }

    private void ClearGridButtonTexts()
    {
        foreach (var text in _gridButtonTextByIndex.Values)
        {
            if (text != null)
            {
                text.text = string.Empty;
            }
        }
    }

    private void SetGridCellText(Vector2Int position, string label)
    {
        if (_gridWidth <= 0 || _gridHeight <= 0)
        {
            return;
        }

        var clamped = new Vector2Int(
            Mathf.Clamp(position.x, 0, _gridWidth - 1),
            Mathf.Clamp(position.y, 0, _gridHeight - 1));

        var index = clamped.y * _gridWidth + clamped.x;
        if (_gridButtonTextByIndex.TryGetValue(index, out var text) && text != null)
        {
            text.text = label ?? string.Empty;
        }
    }

    private void MarkPlayerCell(Vector2Int playerPosition)
    {
        SetGridCellText(playerPosition, PlayerCellLabel);
    }

    private void MarkNpcCellFromDecision(Vector2Int playerPosition, Vector2Int npcPosition)
    {
        if (npcPosition == playerPosition)
        {
            var overlapLabel = string.IsNullOrWhiteSpace(OverlapCellLabel)
                ? $"{PlayerCellLabel} + {NpcCellLabel}"
                : OverlapCellLabel;
            SetGridCellText(playerPosition, overlapLabel);
            return;
        }

        SetGridCellText(npcPosition, NpcCellLabel);
    }

    private async UniTask EnsureSystemPromptAppliedAsync(string systemPrompt, CancellationToken cancel)
    {
        if (ClearHistoryPerPrompt || !_sessionPrimedWithSystemPrompt)
        {
            await Runtime.ClearActiveChatHistoryAsync(systemPrompt, cancel);
            _sessionPrimedWithSystemPrompt = true;
            _lastAppliedSystemPrompt = systemPrompt ?? string.Empty;
            return;
        }

        if (!string.Equals(_lastAppliedSystemPrompt, systemPrompt ?? string.Empty, StringComparison.Ordinal))
        {
            Debug.Log("System prompt changed but ClearHistoryPerPrompt is disabled, so current session prompt remains active.");
        }
    }

    private async UniTask<string> GenerateCompletionAsync(
        string userPrompt,
        InferenceParams inferenceParams,
        Func<string, bool> stopPredicate,
        CancellationToken cancel)
    {
        var completion = new StringBuilder(256);
        await foreach (var token in Runtime.ChatAsync(userPrompt, inferenceParams, cancel))
        {
            completion.Append(token);
            if (stopPredicate != null && stopPredicate(completion.ToString()))
            {
                break;
            }
        }

        return completion.ToString().Trim();
    }

    private async UniTaskVoid HandleGridClick(int buttonIndex)
    {
        var cancel = this.GetCancellationTokenOnDestroy();
        if (!await EnsureRuntimeInitializedAsync(cancel))
        {
            return;
        }

        try
        {
            var ping = new Vector2Int(buttonIndex % _gridWidth, buttonIndex / _gridWidth);
            var request = new LLamaNpcGrammarPlanner.NpcDecisionRequest
            {
                NpcGrid = NpcGrid,
                PlayerPingGrid = ping,
                Behavior = Behavior,
                GridWidth = _gridWidth,
                GridHeight = _gridHeight,
                NearRadius = NearRadius
            };

            Planner.BuildPrompts(request, out var normalizedRequest, out var systemPrompt, out var userPrompt);
            ClearGridButtonTexts();
            MarkPlayerCell(normalizedRequest.PlayerPingGrid);

            SafeLLamaGrammarHandle grammarHandle = null;
            try
            {
                var inferenceParams = Planner.BuildInferenceParams(normalizedRequest, out grammarHandle, out var stopPredicate);
                await EnsureSystemPromptAppliedAsync(systemPrompt, cancel);
                var completion = await GenerateCompletionAsync(userPrompt, inferenceParams, stopPredicate, cancel);
                var trace = Planner.BuildDecisionTrace(normalizedRequest, systemPrompt, userPrompt, completion);

                var npcTarget = new Vector2Int(trace.decision.target_x, trace.decision.target_y);
                NpcGrid = npcTarget;
                MarkNpcCellFromDecision(normalizedRequest.PlayerPingGrid, npcTarget);

                if (SystemPromptOutput != null)
                {
                    SystemPromptOutput.text = trace.system_prompt;
                }

                if (UserPromptOutput != null)
                {
                    UserPromptOutput.text = trace.user_prompt;
                }

                if (CompletionOutput != null)
                {
                    CompletionOutput.text = trace.completion;
                }

                if (DecisionOutput != null)
                {
                    DecisionOutput.text = $"behavior={Behavior}, target=({trace.decision.target_x},{trace.decision.target_y})";
                }

                Debug.Log($"[NPC Planner] Behavior={Behavior}");
                Debug.Log($"[NPC Planner] System Prompt:\n{trace.system_prompt}");
                Debug.Log($"[NPC Planner] User Prompt:\n{trace.user_prompt}");
                Debug.Log($"[NPC Planner] Completion:\n{trace.completion}");
            }
            finally
            {
                grammarHandle?.Dispose();
            }
        }
        catch (Exception ex)
        {
            Debug.LogException(ex);
        }
    }
}
