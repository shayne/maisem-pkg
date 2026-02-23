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
