using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Cysharp.Threading.Tasks;
using System;
using System.Collections.Generic;
using System.Threading;

public class LLamaGridPingController : MonoBehaviour
{
    public LLamaNpcGrammarPlanner Planner;
    public GridLayoutGroup Grid;

    [Header("Runtime Init")]
    public bool AutoInitializeRuntime = true;
    public int RuntimeSessionCount = 1;
    [TextArea(2, 5)]
    public string RuntimeSystemPrompt = "You are a local inference runtime.";

    [Header("NPC")]
    public Vector2Int NpcGrid = new Vector2Int(0, 0);
    public LLamaNpcGrammarPlanner.NpcBehavior Behavior = LLamaNpcGrammarPlanner.NpcBehavior.Cautious;
    public int NearRadius = 3;

    [Header("Output")]
    public TMP_Text SystemPromptOutput;
    public TMP_Text UserPromptOutput;
    public TMP_Text CompletionOutput;
    public TMP_Text DecisionOutput;

    private int _gridWidth;
    private int _gridHeight;
    private readonly List<Button> _gridButtons = new List<Button>();

    private void Awake()
    {
        if (Planner == null)
        {
            Planner = GetComponent<LLamaNpcGrammarPlanner>();
        }

        if (Planner != null && Planner.Runtime == null)
        {
            Planner.Runtime = GetComponent<LLamaModelRuntime>();
        }

        BindButtons();
    }

    private async void Start()
    {
        if (!AutoInitializeRuntime)
        {
            return;
        }

        await EnsureRuntimeInitializedAsync(this.GetCancellationTokenOnDestroy());
    }

    private void BindButtons()
    {
        _gridButtons.Clear();

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
            var index = i;
            button.onClick.RemoveAllListeners();
            button.onClick.AddListener(() => HandleGridClick(index).Forget());
        }
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

        if (Planner.Runtime == null)
        {
            Planner.Runtime = GetComponent<LLamaModelRuntime>();
        }

        if (Planner.Runtime == null)
        {
            Debug.LogError("LLamaGridPingController could not find LLamaModelRuntime.");
            return false;
        }

        if (Planner.Runtime.IsInitialized)
        {
            return true;
        }

        SetGridInteractable(false);
        try
        {
            await Planner.Runtime.InitializeAsync(Mathf.Max(1, RuntimeSessionCount), RuntimeSystemPrompt, cancel);
            return true;
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to initialize LLama runtime for grid planner: {ex}");
            return false;
        }
        finally
        {
            SetGridInteractable(Planner.Runtime.IsInitialized);
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

    private async UniTaskVoid HandleGridClick(int buttonIndex)
    {
        var cancel = this.GetCancellationTokenOnDestroy();
        if (!await EnsureRuntimeInitializedAsync(cancel))
        {
            return;
        }

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

        var trace = await Planner.DecideNpcGridWithTraceAsync(request, cancel);

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
            DecisionOutput.text = $"action={trace.decision.action}, target=({trace.decision.target_x},{trace.decision.target_y})";
        }

        Debug.Log($"[NPC Planner] System Prompt:\n{trace.system_prompt}");
        Debug.Log($"[NPC Planner] User Prompt:\n{trace.user_prompt}");
        Debug.Log($"[NPC Planner] Completion:\n{trace.completion}");
    }
}