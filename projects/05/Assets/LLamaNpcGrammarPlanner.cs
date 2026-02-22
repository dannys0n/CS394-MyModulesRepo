using UnityEngine;
using UnityEngine.Serialization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using LLama.Common;
using LLama.Grammars;
using LLama.Native;

public class LLamaNpcGrammarPlanner : MonoBehaviour
{
    [FormerlySerializedAs("PlannerSystemPrompt")]
    [TextArea(2, 8)]
    public string PlannerBaseInstruction = "You are an NPC tactical planner for a grid game. Respond with exactly one JSON object.";

    [Header("Generation")]
    public float Temperature = 0.15f;
    public int MaxTokens = 64;
    public float RepeatPenalty = 1.05f;
    public int RepeatLastTokensCount = 32;
    public bool TrimToFirstJsonObject = true;

    [Obsolete("Native grammar is always attempted; this compatibility property is ignored.")]
    public bool UseNativeGrammar => true;

    public enum NpcBehavior
    {
        Guard,
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

    public void BuildPrompts(NpcDecisionRequest request, out NpcDecisionRequest normalizedRequest, out string systemPrompt, out string userPrompt)
    {
        normalizedRequest = Normalize(request);
        systemPrompt = BuildSystemPrompt(normalizedRequest);
        userPrompt = BuildUserPrompt(normalizedRequest);
    }

    public InferenceParams BuildInferenceParams(
        NpcDecisionRequest request,
        out SafeLLamaGrammarHandle grammarHandle,
        out Func<string, bool> stopPredicate)
    {
        var normalizedRequest = Normalize(request);

        grammarHandle = null;
        try
        {
            var grammar = Grammar.Parse(BuildDecisionGrammarGbnf(normalizedRequest), "root");
            grammarHandle = grammar.CreateInstance();
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"Native grammar setup failed; falling back to prompt-only JSON constraints. {ex.GetType().Name}: {ex.Message}");
            grammarHandle = null;
        }

        stopPredicate = null;
        if (TrimToFirstJsonObject)
        {
            stopPredicate = text => TryExtractFirstJsonObject(text, out _);
        }

        return new InferenceParams
        {
            Temperature = Temperature,
            MaxTokens = MaxTokens,
            RepeatPenalty = RepeatPenalty,
            RepeatLastTokensCount = RepeatLastTokensCount,
            Grammar = grammarHandle,
            AntiPrompts = new List<string> { "<|im_end|>" }
        };
    }

    [Obsolete("Native grammar is always attempted; the useNativeGrammar argument is ignored.")]
    public InferenceParams BuildInferenceParams(
        NpcDecisionRequest request,
        bool useNativeGrammar,
        out SafeLLamaGrammarHandle grammarHandle,
        out Func<string, bool> stopPredicate)
    {
        return BuildInferenceParams(request, out grammarHandle, out stopPredicate);
    }

    public NpcDecisionTrace BuildDecisionTrace(NpcDecisionRequest request, string systemPrompt, string userPrompt, string completion)
    {
        var normalizedRequest = Normalize(request);
        var normalizedCompletion = string.IsNullOrWhiteSpace(completion) ? string.Empty : completion.Trim();

        if (TrimToFirstJsonObject && TryExtractFirstJsonObject(normalizedCompletion, out var extractedJson))
        {
            normalizedCompletion = extractedJson;
        }

        if (!TryParseDecision(normalizedCompletion, normalizedRequest, out var decision))
        {
            Debug.LogWarning($"NPC planner parse failed. Raw output: {normalizedCompletion}");
            decision = BuildFallbackDecision(normalizedRequest);
        }

        return new NpcDecisionTrace
        {
            system_prompt = systemPrompt,
            user_prompt = userPrompt,
            completion = normalizedCompletion,
            decision = decision
        };
    }

    private string BuildSystemPrompt(NpcDecisionRequest request)
    {
        var baseInstruction = string.IsNullOrWhiteSpace(PlannerBaseInstruction)
            ? "You are an NPC tactical planner."
            : PlannerBaseInstruction.Trim();

        var prompt =
            $"{baseInstruction}\n" +
            "Choose one NPC target position.\n" +
            $"Grid width={request.GridWidth}, height={request.GridHeight}.\n" +
            "Rules:\n" +
            "- aggressive always teleport to same location as ping.\n" +
            "- guard always holds its position.\n" +
            "- scout always moves close to ping but always farther than near radius from ping.\n" +
            "- target_x and target_y must be inside the grid.\n" +
            "User message contains NPC state and player ping coordinates.\n" +
            "Respond with JSON only.";

        prompt +=
            "\nJSON format:\n" +
            "{\"target_x\":<int>,\"target_y\":<int>}\n" +
            "- Do not add markdown, code fences, or extra keys.\n" +
            "- target_x must be an integer in [0, grid width - 1].\n" +
            "- target_y must be an integer in [0, grid height - 1].";

        return prompt;
    }

    private static string BuildUserPrompt(NpcDecisionRequest request)
    {
        var behavior = request.Behavior.ToString().ToLowerInvariant();
        return
            $"NPC previous location x={request.NpcGrid.x}, y={request.NpcGrid.y}.\n" +
            $"NPC behavior={behavior}. Near radius={request.NearRadius}.\n" +
            $"ping_x={request.PlayerPingGrid.x}, ping_y={request.PlayerPingGrid.y}";
    }

    private static NpcDecision BuildFallbackDecision(NpcDecisionRequest request)
    {
        return new NpcDecision
        {
            target_x = request.NpcGrid.x,
            target_y = request.NpcGrid.y
        };
    }

    private static string BuildDecisionGrammarGbnf(NpcDecisionRequest request)
    {
        // Avoid character classes/ranges for older native backends; use explicit literals only.
        var xValues = string.Join(" | ", Enumerable.Range(0, request.GridWidth).Select(v => $"\"{v}\""));
        var yValues = string.Join(" | ", Enumerable.Range(0, request.GridHeight).Select(v => $"\"{v}\""));

        return
            "root ::= \"{\" \"\\\"target_x\\\"\" \":\" x \",\" \"\\\"target_y\\\"\" \":\" y \"}\"\n" +
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

    private static bool TryParseDecision(string json, NpcDecisionRequest request, out NpcDecision decision)
    {
        decision = default;
        if (string.IsNullOrWhiteSpace(json))
        {
            return false;
        }

        try
        {
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
            {
                return false;
            }

            var hasTargetX = false;
            var hasTargetY = false;
            var targetX = 0;
            var targetY = 0;
            var propertyCount = 0;
            foreach (var property in doc.RootElement.EnumerateObject())
            {
                propertyCount++;
                if (property.NameEquals("target_x"))
                {
                    if (hasTargetX)
                    {
                        return false;
                    }

                    if (property.Value.ValueKind != JsonValueKind.Number || !property.Value.TryGetInt32(out targetX))
                    {
                        return false;
                    }

                    hasTargetX = true;
                    continue;
                }

                if (property.NameEquals("target_y"))
                {
                    if (hasTargetY)
                    {
                        return false;
                    }

                    if (property.Value.ValueKind != JsonValueKind.Number || !property.Value.TryGetInt32(out targetY))
                    {
                        return false;
                    }

                    hasTargetY = true;
                    continue;
                }

                return false;
            }

            if (!hasTargetX || !hasTargetY || propertyCount != 2)
            {
                return false;
            }

            if (targetX < 0 || targetX >= request.GridWidth || targetY < 0 || targetY >= request.GridHeight)
            {
                return false;
            }

            decision = new NpcDecision
            {
                target_x = targetX,
                target_y = targetY
            };
        }
        catch
        {
            return false;
        }

        if (request.Behavior == NpcBehavior.Guard)
        {
            decision.target_x = request.NpcGrid.x;
            decision.target_y = request.NpcGrid.y;
            return true;
        }

        if (request.Behavior == NpcBehavior.Aggressive)
        {
            decision.target_x = request.PlayerPingGrid.x;
            decision.target_y = request.PlayerPingGrid.y;
            return true;
        }

        if (request.Behavior == NpcBehavior.Scout)
        {
            var target = new Vector2Int(decision.target_x, decision.target_y);
            var pingDistance = ManhattanDistance(target, request.PlayerPingGrid);
            var desiredScoutDistance = request.NearRadius + 1;
            if (pingDistance != desiredScoutDistance)
            {
                if (!TryFindClosestPointOutsidePingRadius(request, desiredScoutDistance, out var correctedTarget))
                {
                    return false;
                }

                decision.target_x = correctedTarget.x;
                decision.target_y = correctedTarget.y;
            }
        }

        return true;
    }

    private static int ManhattanDistance(Vector2Int a, Vector2Int b)
    {
        return Mathf.Abs(a.x - b.x) + Mathf.Abs(a.y - b.y);
    }

    private static bool TryFindClosestPointOutsidePingRadius(NpcDecisionRequest request, int desiredPingDistance, out Vector2Int target)
    {
        var found = false;
        var bestPingDistanceDelta = int.MaxValue;
        var bestNpcDistance = int.MaxValue;
        target = request.NpcGrid;

        for (var y = 0; y < request.GridHeight; y++)
        {
            for (var x = 0; x < request.GridWidth; x++)
            {
                var point = new Vector2Int(x, y);
                var pingDistance = ManhattanDistance(point, request.PlayerPingGrid);
                if (pingDistance <= request.NearRadius)
                {
                    continue;
                }

                var pingDistanceDelta = Mathf.Abs(pingDistance - desiredPingDistance);
                var npcDistance = ManhattanDistance(point, request.NpcGrid);
                if (pingDistanceDelta < bestPingDistanceDelta ||
                    (pingDistanceDelta == bestPingDistanceDelta && npcDistance < bestNpcDistance))
                {
                    found = true;
                    bestPingDistanceDelta = pingDistanceDelta;
                    bestNpcDistance = npcDistance;
                    target = point;
                }
            }
        }

        return found;
    }
}
