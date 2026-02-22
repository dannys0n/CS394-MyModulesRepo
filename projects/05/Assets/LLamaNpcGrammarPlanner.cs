using UnityEngine;
using System;
using System.Text.Json;
using System.Threading;
using Cysharp.Threading.Tasks;
using LLama.Common;
using LLama.Grammars;

public class LLamaNpcGrammarPlanner : MonoBehaviour
{
    public LLamaModelRuntime Runtime;

    [TextArea(2, 6)]
    public string PlannerSystemPrompt = "You are an NPC tactical planner for a grid game. Respond with exactly one JSON object.";

    public float Temperature = 0.15f;
    public int MaxTokens = 64;
    public float RepeatPenalty = 1.05f;
    public int RepeatLastTokensCount = 32;

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

    [Serializable]
    public struct NpcDecisionTrace
    {
        public string system_prompt;
        public string user_prompt;
        public string completion;
        public NpcDecision decision;
    }

    private const string NpcDecisionGrammarGbnf =
        "root ::= ws \"{\" ws \"\\\"action\\\"\" ws \":\" ws action ws \",\" ws \"\\\"target_x\\\"\" ws \":\" ws int ws \",\" ws \"\\\"target_y\\\"\" ws \":\" ws int ws \"}\" ws\n" +
        "action ::= \"\\\"hold\\\"\" | \"\\\"move_to_ping\\\"\" | \"\\\"move_near_self\\\"\"\n" +
        "int ::= \"-\"? [0-9] [0-9]*\n" +
        "ws ::= [ \\t\\n\\r]*\n";

    private static readonly Grammar NpcDecisionGrammar = Grammar.Parse(NpcDecisionGrammarGbnf, "root");
    private static readonly JsonSerializerOptions JsonOptions = new JsonSerializerOptions { IncludeFields = true };

    public async UniTask<NpcDecision> DecideNpcGridAsync(NpcDecisionRequest request, CancellationToken cancel = default)
    {
        var trace = await DecideNpcGridWithTraceAsync(request, cancel);
        return trace.decision;
    }

    public async UniTask<NpcDecisionTrace> DecideNpcGridWithTraceAsync(NpcDecisionRequest request, CancellationToken cancel = default)
    {
        if (Runtime == null)
        {
            throw new InvalidOperationException("LLamaNpcGrammarPlanner requires a LLamaModelRuntime reference.");
        }

        request = Normalize(request);
        var userPrompt = BuildPrompt(request);
        using var grammarHandle = NpcDecisionGrammar.CreateInstance();

        var inferenceParams = new InferenceParams
        {
            Temperature = Temperature,
            MaxTokens = MaxTokens,
            RepeatPenalty = RepeatPenalty,
            RepeatLastTokensCount = RepeatLastTokensCount,
            Grammar = grammarHandle
        };

        var completion = await Runtime.CompleteOnceAsync(PlannerSystemPrompt, userPrompt, inferenceParams, cancel);

        NpcDecision decision;
        if (!TryParseDecision(completion, request, out decision))
        {
            Debug.LogWarning($"NPC grammar parse failed. Raw output: {completion}");
            decision = new NpcDecision
            {
                action = "hold",
                target_x = request.NpcGrid.x,
                target_y = request.NpcGrid.y
            };
        }

        return new NpcDecisionTrace
        {
            system_prompt = PlannerSystemPrompt,
            user_prompt = userPrompt,
            completion = completion,
            decision = decision
        };
    }

    private static NpcDecisionRequest Normalize(NpcDecisionRequest request)
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

    private static string BuildPrompt(NpcDecisionRequest request)
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

    private static bool TryParseDecision(string json, NpcDecisionRequest request, out NpcDecision decision)
    {
        decision = default;
        if (string.IsNullOrWhiteSpace(json))
        {
            return false;
        }

        try
        {
            decision = JsonSerializer.Deserialize<NpcDecision>(json, JsonOptions);
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
}