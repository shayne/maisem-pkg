// Copyright (c) 2025 AUTHORS All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/openai/openai-go"
	"pkg.maisem.dev/agent"
)

func paramsAsMap(t *testing.T, params any) map[string]any {
	t.Helper()

	b, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	return out
}

func TestBuildChatCompletionParams_GPT5FamilyUsesMaxCompletionTokens(t *testing.T) {
	client := &Client{model: "gpt-5.3-codex"}
	params := client.buildChatCompletionParams(agent.MessagesRequest{MaxTokens: 321})
	asMap := paramsAsMap(t, params)

	got, ok := asMap["max_completion_tokens"]
	if !ok {
		t.Fatalf("expected max_completion_tokens to be present, payload=%v", asMap)
	}
	if int(got.(float64)) != 321 {
		t.Fatalf("max_completion_tokens mismatch: got=%v want=321", got)
	}
	if _, ok := asMap["max_tokens"]; ok {
		t.Fatalf("expected max_tokens to be omitted for GPT-5.x models, payload=%v", asMap)
	}
}

func TestBuildChatCompletionParams_NonGPT5UsesMaxTokens(t *testing.T) {
	client := &Client{model: "gpt-4.1-mini"}
	params := client.buildChatCompletionParams(agent.MessagesRequest{MaxTokens: 123})
	asMap := paramsAsMap(t, params)

	got, ok := asMap["max_tokens"]
	if !ok {
		t.Fatalf("expected max_tokens to be present, payload=%v", asMap)
	}
	if int(got.(float64)) != 123 {
		t.Fatalf("max_tokens mismatch: got=%v want=123", got)
	}
	if _, ok := asMap["max_completion_tokens"]; ok {
		t.Fatalf("expected max_completion_tokens to be omitted for non-GPT-5 models, payload=%v", asMap)
	}
}

func TestBuildResponsesParams_ToolsDoNotForceStrictSchema(t *testing.T) {
	client := &Client{model: "gpt-5.3-codex"}
	params := client.buildResponsesParams(agent.MessagesRequest{
		Tools: []agent.ToolDefinition{{
			Name:        "create-reminder",
			Description: "Create a reminder",
			InputSchema: json.RawMessage(`{
				"type": "object",
				"properties": {
					"message": {"type": "string"},
					"requires_acknowledgement": {"type": "boolean"}
				},
				"required": ["message"]
			}`),
		}},
	})
	asMap := paramsAsMap(t, params)

	tools, ok := asMap["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("expected one tool in responses params, got %#v", asMap["tools"])
	}
	tool, ok := tools[0].(map[string]any)
	if !ok {
		t.Fatalf("expected tool entry to be a map, got %#v", tools[0])
	}
	if got, has := tool["strict"]; has && got == true {
		t.Fatalf("expected strict schema to be disabled for responses tools, got strict=%v payload=%v", got, tool)
	}
}

func TestBuildResponsesParams_UsesPreviousResponseID(t *testing.T) {
	client := &Client{model: "gpt-5.3-codex"}
	params := client.buildResponsesParams(agent.MessagesRequest{
		ConversationState: &agent.ConversationState{
			PreviousResponseID: "  resp_123  ",
		},
	})
	asMap := paramsAsMap(t, params)

	got, ok := asMap["previous_response_id"]
	if !ok {
		t.Fatalf("expected previous_response_id to be present, payload=%v", asMap)
	}
	if got != "resp_123" {
		t.Fatalf("previous_response_id mismatch: got=%v want=resp_123", got)
	}
}

func TestBuildResponsesParams_OmitsPreviousResponseIDWhenEmpty(t *testing.T) {
	tests := []struct {
		name string
		req  agent.MessagesRequest
	}{
		{
			name: "missing conversation state",
			req:  agent.MessagesRequest{},
		},
		{
			name: "empty previous response id",
			req: agent.MessagesRequest{
				ConversationState: &agent.ConversationState{},
			},
		},
		{
			name: "whitespace previous response id",
			req: agent.MessagesRequest{
				ConversationState: &agent.ConversationState{
					PreviousResponseID: "   ",
				},
			},
		},
	}

	client := &Client{model: "gpt-5.3-codex"}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params := client.buildResponsesParams(tt.req)
			asMap := paramsAsMap(t, params)
			if got, ok := asMap["previous_response_id"]; ok {
				t.Fatalf("expected previous_response_id to be omitted, got=%v payload=%v", got, asMap)
			}
		})
	}
}

func TestNormalizeToolCallArguments_EmptyDefaultsToObject(t *testing.T) {
	got := normalizeToolCallArguments("")
	if string(got) != "{}" {
		t.Fatalf("expected empty args to normalize to {}, got %q", string(got))
	}
}

func TestNormalizeToolCallArguments_InvalidFallsBackToJSONString(t *testing.T) {
	got := normalizeToolCallArguments("{")
	if !json.Valid(got) {
		t.Fatalf("expected normalized args to be valid JSON, got %q", string(got))
	}
	if len(got) == 0 || got[0] != '"' {
		t.Fatalf("expected invalid args to become JSON string literal, got %q", string(got))
	}
}

func TestCollectAndBuildToolUseContents_AccumulatesArgumentsAcrossChunks(t *testing.T) {
	byIndex := map[int64]*streamedToolCall{}
	var order []int64

	collectStreamToolCalls([]openai.ChatCompletionChunkChoiceDeltaToolCall{
		{
			Index: 0,
			ID:    "call_1",
			Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
				Name:      "create-reminder",
				Arguments: "",
			},
		},
	}, byIndex, &order)

	collectStreamToolCalls([]openai.ChatCompletionChunkChoiceDeltaToolCall{
		{
			Index: 0,
			Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
				Arguments: `{"message":"hi"`,
			},
		},
	}, byIndex, &order)

	collectStreamToolCalls([]openai.ChatCompletionChunkChoiceDeltaToolCall{
		{
			Index: 0,
			Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
				Arguments: `,"reminder_at":"2026-02-23T12:00:00Z"}`,
			},
		},
	}, byIndex, &order)

	contents := buildToolUseContents(byIndex, order)
	if len(contents) != 1 {
		t.Fatalf("expected 1 tool use content, got %d", len(contents))
	}
	got := contents[0]
	if got.Type != agent.ContentTypeToolUse || got.ToolUse == nil {
		t.Fatalf("expected tool use content, got %#v", got)
	}
	if got.ToolUse.ID != "call_1" {
		t.Fatalf("tool call id mismatch: got %q", got.ToolUse.ID)
	}
	if got.ToolUse.Name != "create-reminder" {
		t.Fatalf("tool call name mismatch: got %q", got.ToolUse.Name)
	}
	if !json.Valid(got.ToolUse.Input) {
		t.Fatalf("expected valid JSON arguments, got %q", string(got.ToolUse.Input))
	}
	if !strings.Contains(string(got.ToolUse.Input), `"message":"hi"`) {
		t.Fatalf("expected merged args to include message, got %q", string(got.ToolUse.Input))
	}
	if !strings.Contains(string(got.ToolUse.Input), `"reminder_at":"2026-02-23T12:00:00Z"`) {
		t.Fatalf("expected merged args to include reminder_at, got %q", string(got.ToolUse.Input))
	}
}

func TestUsesResponsesAPI_CodexModels(t *testing.T) {
	if !usesResponsesAPI("gpt-5.3-codex") {
		t.Fatalf("expected codex model to use responses API")
	}
	if usesResponsesAPI("gpt-5.2") {
		t.Fatalf("expected non-codex model to use chat completions API")
	}
}

func TestConvertToResponsesInput_AssistantTextUsesOutputText(t *testing.T) {
	input := convertToResponsesInput([]agent.Message{
		{
			Role: agent.RoleAssistant,
			Content: agent.Content{
				agent.NewTextContent("hello from assistant history"),
			},
		},
	})

	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	var items []map[string]any
	if err := json.Unmarshal(raw, &items); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 input item, got %d (%s)", len(items), string(raw))
	}

	content, ok := items[0]["content"].([]any)
	if !ok || len(content) != 1 {
		t.Fatalf("expected one content part, got %#v", items[0]["content"])
	}
	part, ok := content[0].(map[string]any)
	if !ok {
		t.Fatalf("expected map content part, got %#v", content[0])
	}
	id, _ := items[0]["id"].(string)
	if !strings.HasPrefix(id, "msg") {
		t.Fatalf("assistant history id must start with msg, got %q (payload=%s)", id, string(raw))
	}
	if got := part["type"]; got != "output_text" {
		t.Fatalf("assistant history must use output_text, got %v (payload=%s)", got, string(raw))
	}
	if got := part["text"]; got != "hello from assistant history" {
		t.Fatalf("assistant text mismatch: got %v", got)
	}
}

func TestConvertToResponsesInput_AssistantIDsAreUnique(t *testing.T) {
	input := convertToResponsesInput([]agent.Message{
		{
			Role:    agent.RoleAssistant,
			Content: agent.Content{agent.NewTextContent("first")},
		},
		{
			Role:    agent.RoleAssistant,
			Content: agent.Content{agent.NewTextContent("second")},
		},
	})

	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	var items []map[string]any
	if err := json.Unmarshal(raw, &items); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("expected 2 input items, got %d (%s)", len(items), string(raw))
	}

	firstID, _ := items[0]["id"].(string)
	secondID, _ := items[1]["id"].(string)
	if firstID == secondID {
		t.Fatalf("assistant message IDs must be unique, both were %q (payload=%s)", firstID, string(raw))
	}
	if !strings.HasPrefix(firstID, "msg") || !strings.HasPrefix(secondID, "msg") {
		t.Fatalf("assistant message IDs must start with msg, got %q and %q", firstID, secondID)
	}
}
