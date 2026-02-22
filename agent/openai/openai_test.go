// Copyright (c) 2025 AUTHORS All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package openai

import (
	"encoding/json"
	"testing"

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
	client := &Client{model: "gpt-5.2"}
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
