using UnityEngine;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading;
using Cysharp.Threading.Tasks;
using LLama.Common;
using LLama.Grammars;
using LLama.Native;

public class LLamaNpcGrammarPlanner : MonoBehaviour
{
    public LLamaModelRuntime Runtime;

    [TextArea(2, 6)]
    public string PlannerSystemPrompt = "You are an NPC tactical planner for a grid game. Respond with exactly one JSON object.";

    [Header("Generation")]
    public float Temperature = 0.15f;
    public int MaxTokens = 64;
    public float RepeatPenalty = 1.05f;
    public int RepeatLastTokensCount = 32;
    public bool UseNativeGrammar = false;
    public bool TrimToFirstJsonObject = true;

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

        SafeLLamaGrammarHandle grammarHandle = null;
        try
        {
            if (UseNativeGrammar)
            {
                var grammar = Grammar.Parse(BuildDecisionGrammarGbnf(request), "root");
                grammarHandle = grammar.CreateInstance();
            }

            var inferenceParams = new InferenceParams
            {
                Temperature = Temperature,
                MaxTokens = MaxTokens,
                RepeatPenalty = RepeatPenalty,
                RepeatLastTokensCount = RepeatLastTokensCount,
                Grammar = grammarHandle,
                AntiPrompts = new List<string> { "<|im_end|>" }
            };

            Func<string, bool> stopPredicate = null;
            if (TrimToFirstJsonObject)
            {
                stopPredicate = text => TryExtractFirstJsonObject(text, out _);
            }

            var completion = await Runtime.CompleteOnceAsync(PlannerSystemPrompt, userPrompt, inferenceParams, cancel, stopPredicate);

            if (TrimToFirstJsonObject && TryExtractFirstJsonObject(completion, out var extractedJson))
            {
                completion = extractedJson;
            }

            NpcDecision decision;
            if (!TryParseDecision(completion, request, out decision))
            {
                Debug.LogWarning($"NPC planner parse failed. Raw output: {completion}");
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
        finally
        {
            grammarHandle?.Dispose();
        }
    }

    private static string BuildDecisionGrammarGbnf(NpcDecisionRequest request)
    {
        // Avoid character classes/ranges for older native backends; use explicit literals only.
        var xValues = string.Join(" | ", Enumerable.Range(0, request.GridWidth).Select(v => $"\"{v}\""));
        var yValues = string.Join(" | ", Enumerable.Range(0, request.GridHeight).Select(v => $"\"{v}\""));

        return
            "root ::= \"{\" \"\\\"action\\\"\" \":\" action \",\" \"\\\"target_x\\\"\" \":\" x \",\" \"\\\"target_y\\\"\" \":\" y \"}\"\n" +
            "action ::= \"\\\"hold\\\"\" | \"\\\"move_to_ping\\\"\" | \"\\\"move_near_self\\\"\"\n" +
            $"x ::= {xValues}\n" +
            $"y ::= {yValues}\n";
    }

    private static bool TryExtractFirstJsonObject(string text, out string json)
    {
        json = null;
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        var start = text.IndexOf('{');
        if (start < 0)
        {
            return false;
        }

        var depth = 0;
        var inString = false;
        var escaped = false;

        for (var i = start; i < text.Length; i++)
        {
            var c = text[i];

            if (inString)
            {
                if (escaped)
                {
                    escaped = false;
                    continue;
                }

                if (c == '\\')
                {
                    escaped = true;
                    continue;
                }

                if (c == '"')
                {
                    inString = false;
                }

                continue;
            }

            if (c == '"')
            {
                inString = true;
                continue;
            }

            if (c == '{')
            {
                depth++;
                continue;
            }

            if (c == '}')
            {
                depth--;
                if (depth == 0)
                {
                    json = text.Substring(start, i - start + 1);
                    return true;
                }
            }
        }

        return false;
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
