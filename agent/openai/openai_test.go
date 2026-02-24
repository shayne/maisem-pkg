// Copyright (c) 2025 AUTHORS All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package openai

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/responses"
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

func responseFromJSON(t *testing.T, raw string) *responses.Response {
	t.Helper()
	var resp responses.Response
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("json.Unmarshal response: %v", err)
	}
	return &resp
}

type loopClient struct {
	t     *testing.T
	calls int
}

func TestCreateMessagesWithResponses_RetriesTopLevelEmptyOutputTextOnlyOnce(t *testing.T) {
	t.Helper()

	callCount := 0
	c := &Client{
		model: "gpt-5.3-codex",
		responsesNewHook: func(ctx context.Context, params responses.ResponseNewParams) (*responses.Response, error) {
			callCount++
			switch callCount {
			case 1:
				return responseFromJSON(t, `{
					"id":"resp_empty_1","object":"response","created_at":0,
					"status":"completed","error":null,"incomplete_details":null,
					"instructions":null,"metadata":{},"model":"gpt-5.3-codex",
					"output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"","annotations":[]}]}],
					"parallel_tool_calls":true,"temperature":1,"tool_choice":"auto","tools":[],"top_p":1,
					"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":0,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":1}
				}`), nil
			case 2:
				return responseFromJSON(t, `{
					"id":"resp_ok_1","object":"response","created_at":0,
					"status":"completed","error":null,"incomplete_details":null,
					"instructions":null,"metadata":{},"model":"gpt-5.3-codex",
					"output":[{"id":"msg_2","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Dublin.","annotations":[]}]}],
					"parallel_tool_calls":true,"temperature":1,"tool_choice":"auto","tools":[],"top_p":1,
					"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}
				}`), nil
			default:
				t.Fatalf("unexpected responses.New call %d", callCount)
				return nil, nil
			}
		},
	}
	c.SetRetryEmptyCompletedMessageOnce(true)

	resp, err := c.createMessagesWithResponses(context.Background(), agent.MessagesRequest{
		Messages: []agent.Message{{
			Role:    agent.RoleUser,
			Content: agent.Content{agent.NewTextContent("What is the capital of Ireland?")},
		}},
	})
	if err != nil {
		t.Fatalf("createMessagesWithResponses error: %v", err)
	}
	if callCount != 2 {
		t.Fatalf("responses.New calls = %d, want 2", callCount)
	}
	if got := resp.Content.LossyText(); got != "Dublin." {
		t.Fatalf("response text = %q, want Dublin.", got)
	}
}

func TestCreateMessagesWithResponses_RetriesTopLevelEmptyOutputTextOnlyUpToConfiguredLimit(t *testing.T) {
	t.Helper()

	callCount := 0
	c := &Client{
		model: "gpt-5.3-codex",
		responsesNewHook: func(ctx context.Context, params responses.ResponseNewParams) (*responses.Response, error) {
			callCount++
			if callCount <= 5 {
				return responseFromJSON(t, `{
					"id":"resp_empty_cfg","object":"response","created_at":0,
					"status":"completed","error":null,"incomplete_details":null,
					"instructions":null,"metadata":{},"model":"gpt-5.3-codex",
					"output":[{"id":"msg_cfg_empty","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"","annotations":[]}]}],
					"parallel_tool_calls":true,"temperature":1,"tool_choice":"auto","tools":[],"top_p":1,
					"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":0,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":1}
				}`), nil
			}
			if callCount == 6 {
				return responseFromJSON(t, `{
					"id":"resp_cfg_ok","object":"response","created_at":0,
					"status":"completed","error":null,"incomplete_details":null,
					"instructions":null,"metadata":{},"model":"gpt-5.3-codex",
					"output":[{"id":"msg_cfg_ok","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Dublin.","annotations":[]}]}],
					"parallel_tool_calls":true,"temperature":1,"tool_choice":"auto","tools":[],"top_p":1,
					"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}
				}`), nil
			}
			t.Fatalf("unexpected responses.New call %d", callCount)
			return nil, nil
		},
	}
	c.SetRetryEmptyCompletedMessageLimit(5)

	resp, err := c.createMessagesWithResponses(context.Background(), agent.MessagesRequest{
		Messages: []agent.Message{{
			Role:    agent.RoleUser,
			Content: agent.Content{agent.NewTextContent("What is the capital of Ireland?")},
		}},
	})
	if err != nil {
		t.Fatalf("createMessagesWithResponses error: %v", err)
	}
	if callCount != 6 {
		t.Fatalf("responses.New calls = %d, want 6 (1 initial + 5 retries)", callCount)
	}
	if got := resp.Content.LossyText(); got != "Dublin." {
		t.Fatalf("response text = %q, want Dublin.", got)
	}
}

func TestCreateMessagesWithResponses_DoesNotRetryContinuationEmptyOutputTextOnlyNoop(t *testing.T) {
	t.Helper()

	callCount := 0
	c := &Client{
		model: "gpt-5.3-codex",
		responsesNewHook: func(ctx context.Context, params responses.ResponseNewParams) (*responses.Response, error) {
			callCount++
			return responseFromJSON(t, `{
				"id":"resp_cont_empty_1","object":"response","created_at":0,
				"status":"completed","error":null,"incomplete_details":null,
				"instructions":null,"metadata":{},"model":"gpt-5.3-codex",
				"output":[{"id":"msg_3","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"","annotations":[]}]}],
				"parallel_tool_calls":true,"temperature":1,"tool_choice":"auto","tools":[],"top_p":1,
				"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":0,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":1}
			}`), nil
		},
	}
	c.SetRetryEmptyCompletedMessageOnce(true)

	resp, err := c.createMessagesWithResponses(context.Background(), agent.MessagesRequest{
		Messages: []agent.Message{{
			Role: agent.RoleUser,
			Content: agent.Content{
				agent.NewTextContent("continue"),
			},
		}},
		ConversationState: &agent.ConversationState{PreviousResponseID: "resp_prev"},
	})
	if err != nil {
		t.Fatalf("createMessagesWithResponses continuation noop error: %v", err)
	}
	if callCount != 1 {
		t.Fatalf("responses.New calls = %d, want 1 for continuation noop", callCount)
	}
	if len(resp.Content) != 0 {
		t.Fatalf("expected no content for continuation noop, got %#v", resp.Content)
	}
}

func TestCreateMessagesWithResponses_EmptyOutputTextOnlyAfterRetryReturnsTypedError(t *testing.T) {
	t.Helper()

	callCount := 0
	c := &Client{
		model: "gpt-5.3-codex",
		responsesNewHook: func(ctx context.Context, params responses.ResponseNewParams) (*responses.Response, error) {
			callCount++
			return responseFromJSON(t, `{
				"id":"resp_empty_repeat","object":"response","created_at":0,
				"status":"completed","error":null,"incomplete_details":null,
				"instructions":null,"metadata":{},"model":"gpt-5.3-codex",
				"output":[{"id":"msg_empty","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"","annotations":[]}]}],
				"parallel_tool_calls":true,"temperature":1,"tool_choice":"auto","tools":[],"top_p":1,
				"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":0,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":1}
			}`), nil
		},
	}
	c.SetRetryEmptyCompletedMessageOnce(true)

	_, err := c.createMessagesWithResponses(context.Background(), agent.MessagesRequest{
		Messages: []agent.Message{{
			Role:    agent.RoleUser,
			Content: agent.Content{agent.NewTextContent("What is the capital of Ireland?")},
		}},
	})
	if err == nil {
		t.Fatalf("expected error after retry exhaustion")
	}
	if !errors.Is(err, ErrResponsesEmptyCompletedMessage) {
		t.Fatalf("error %v is not ErrResponsesEmptyCompletedMessage", err)
	}
	if callCount != 2 {
		t.Fatalf("responses.New calls = %d, want 2 after retry exhaustion", callCount)
	}
}

func TestCreateMessagesWithResponses_DefaultDoesNotRetryTopLevelEmptyOutputTextOnly(t *testing.T) {
	t.Helper()

	callCount := 0
	c := &Client{
		model: "gpt-5.3-codex",
		responsesNewHook: func(ctx context.Context, params responses.ResponseNewParams) (*responses.Response, error) {
			callCount++
			return responseFromJSON(t, `{
				"id":"resp_empty_default","object":"response","created_at":0,
				"status":"completed","error":null,"incomplete_details":null,
				"instructions":null,"metadata":{},"model":"gpt-5.3-codex",
				"output":[{"id":"msg_empty_default","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"","annotations":[]}]}],
				"parallel_tool_calls":true,"temperature":1,"tool_choice":"auto","tools":[],"top_p":1,
				"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":0,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":1}
			}`), nil
		},
	}

	_, err := c.createMessagesWithResponses(context.Background(), agent.MessagesRequest{
		Messages: []agent.Message{{
			Role:    agent.RoleUser,
			Content: agent.Content{agent.NewTextContent("What is the capital of Ireland?")},
		}},
	})
	if err == nil {
		t.Fatalf("expected error without retry setting enabled")
	}
	if !errors.Is(err, ErrResponsesEmptyCompletedMessage) {
		t.Fatalf("error %v is not ErrResponsesEmptyCompletedMessage", err)
	}
	if callCount != 1 {
		t.Fatalf("responses.New calls = %d, want 1 when retry setting is disabled", callCount)
	}
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

func TestClientSupportsPreviousResponseID(t *testing.T) {
	if !(&Client{model: "gpt-5.3-codex"}).SupportsPreviousResponseID() {
		t.Fatalf("expected codex/responses model to support previous_response_id")
	}
	if (&Client{model: "gpt-4.1"}).SupportsPreviousResponseID() {
		t.Fatalf("expected chat-completions model to not support previous_response_id continuation")
	}
}

func TestResponsesRequestStateMode(t *testing.T) {
	tests := []struct {
		name string
		req  agent.MessagesRequest
		want string
	}{
		{
			name: "manual replay",
			req: agent.MessagesRequest{
				Messages: []agent.Message{{Role: agent.RoleUser, Content: agent.Content{agent.NewTextContent("hi")}}},
			},
			want: "manual_replay",
		},
		{
			name: "previous response continuation with delta input",
			req: agent.MessagesRequest{
				ConversationState: &agent.ConversationState{PreviousResponseID: "resp_1"},
				Messages:          []agent.Message{{Role: agent.RoleUser, Content: agent.Content{agent.NewTextContent("next")}}},
			},
			want: "previous_response_id_continuation",
		},
		{
			name: "empty stateless request",
			req:  agent.MessagesRequest{},
			want: "stateless_empty",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := responsesRequestStateMode(tt.req); got != tt.want {
				t.Fatalf("responsesRequestStateMode() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestResponseOutputToContentReportsUnhandledItemTypes(t *testing.T) {
	var output []responses.ResponseOutputItemUnion
	if err := json.Unmarshal([]byte(`[
		{
			"id":"ctc_123",
			"type":"custom_tool_call",
			"status":"completed",
			"call_id":"call_custom_1",
			"input":"echo hello",
			"name":"shell_exec"
		},
		{
			"id":"msg_123",
			"type":"message",
			"status":"completed",
			"role":"assistant",
			"content":[{"type":"output_text","text":"done","annotations":[]}]
		}
	]`), &output); err != nil {
		t.Fatalf("json.Unmarshal output items: %v", err)
	}

	content, dropped, ignoredParts := responseOutputToContentWithUnhandled(output)
	if got := len(content); got != 1 {
		t.Fatalf("content item count = %d, want 1", got)
	}
	if content[0].Type != agent.ContentTypeText || content[0].Text != "done" {
		t.Fatalf("content[0] = %#v, want assistant text", content[0])
	}
	if got := dropped["custom_tool_call"]; got != 1 {
		t.Fatalf("dropped custom_tool_call count = %d, want 1 (dropped=%v)", got, dropped)
	}
	if len(ignoredParts) != 0 {
		t.Fatalf("ignored message parts = %v, want none", ignoredParts)
	}
}

func TestResponseOutputToContentConvertsRefusalMessagePartToText(t *testing.T) {
	var output []responses.ResponseOutputItemUnion
	if err := json.Unmarshal([]byte(`[
		{
			"id":"msg_refusal_1",
			"type":"message",
			"status":"completed",
			"role":"assistant",
			"content":[{"type":"refusal","refusal":"I can’t help with that."}]
		}
	]`), &output); err != nil {
		t.Fatalf("json.Unmarshal output items: %v", err)
	}

	content, dropped, ignoredParts := responseOutputToContentWithUnhandled(output)
	if got := len(content); got != 1 {
		t.Fatalf("content item count = %d, want 1 for refusal-only message", got)
	}
	if content[0].Type != agent.ContentTypeText {
		t.Fatalf("content[0].Type = %q, want text", content[0].Type)
	}
	if content[0].Text != "I can’t help with that." {
		t.Fatalf("content[0].Text = %q, want refusal text", content[0].Text)
	}
	if len(dropped) != 0 {
		t.Fatalf("dropped output item types = %v, want none", dropped)
	}
	if len(ignoredParts) != 0 {
		t.Fatalf("ignored message parts = %v, want none", ignoredParts)
	}
}

func TestUnsupportedResponsesNoContentError_EmptyOutputTextOnlyRequiresContinuationFlag(t *testing.T) {
	var output []responses.ResponseOutputItemUnion
	if err := json.Unmarshal([]byte(`[
		{
			"id":"msg_empty_1",
			"type":"message",
			"status":"completed",
			"role":"assistant",
			"content":[{"type":"output_text","text":"","annotations":[]}]
		}
	]`), &output); err != nil {
		t.Fatalf("json.Unmarshal output items: %v", err)
	}

	content, dropped, ignoredParts := responseOutputToContentWithUnhandled(output)
	if got := len(content); got != 0 {
		t.Fatalf("content item count = %d, want 0 for empty output_text-only message", got)
	}
	if len(dropped) != 0 {
		t.Fatalf("dropped output item types = %v, want none", dropped)
	}
	if got := ignoredParts["output_text_empty"]; got != 1 {
		t.Fatalf("ignored empty output_text part count = %d, want 1 (ignored=%v)", got, ignoredParts)
	}

	resp := responses.Response{
		Status: responses.ResponseStatusCompleted,
		Output: output,
	}
	if err := unsupportedResponsesNoContentError(&resp, dropped, ignoredParts, false); err == nil {
		t.Fatalf("expected top-level empty output_text-only message to remain an error without continuation flag")
	}
	if err := unsupportedResponsesNoContentError(&resp, dropped, ignoredParts, true); err != nil {
		t.Fatalf("expected continuation empty output_text-only message to be treated as benign no-op, got error: %v", err)
	}
}

func TestUnsupportedResponsesNoContentError_IncludesResponseStatusDiagnostics(t *testing.T) {
	var resp responses.Response
	if err := json.Unmarshal([]byte(`{
		"id":"resp_test_1",
		"object":"response",
		"created_at":0,
		"status":"completed",
		"error": null,
		"incomplete_details": null,
		"instructions": null,
		"metadata": {},
		"model":"gpt-5.3-codex",
		"output":[
			{
				"id":"msg_empty_2",
				"type":"message",
				"status":"completed",
				"role":"assistant",
				"content":[{"type":"output_text","text":"","annotations":[]}]
			}
		],
		"parallel_tool_calls": true,
		"temperature": 1,
		"tool_choice":"auto",
		"tools":[],
		"top_p":1
	}`), &resp); err != nil {
		t.Fatalf("json.Unmarshal response: %v", err)
	}

	content, dropped, ignoredParts := responseOutputToContentWithUnhandled(resp.Output)
	if got := len(content); got != 0 {
		t.Fatalf("content item count = %d, want 0", got)
	}

	err := unsupportedResponsesNoContentError(&resp, dropped, ignoredParts, false)
	if err == nil {
		t.Fatalf("expected unsupported no-content error")
	}
	if !strings.Contains(err.Error(), "response_status=completed") {
		t.Fatalf("error %q missing response_status diagnostics", err)
	}
	if !strings.Contains(err.Error(), "output_item_statuses=message:completed=1") {
		t.Fatalf("error %q missing output_item_statuses diagnostics", err)
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

func TestResponsesReasoningItemsArePreservedForManualReplay(t *testing.T) {
	var output []responses.ResponseOutputItemUnion
	if err := json.Unmarshal([]byte(`[
		{
			"id":"rs_123",
			"type":"reasoning",
			"summary":[{"type":"summary_text","text":"Need weather data first."}],
			"status":"completed"
		},
		{
			"id":"fc_123",
			"type":"function_call",
			"call_id":"call_weather_1",
			"name":"get_weather",
			"arguments":"{\"location\":\"Dublin\"}",
			"status":"completed"
		}
	]`), &output); err != nil {
		t.Fatalf("json.Unmarshal output items: %v", err)
	}

	content := responseOutputToContent(output)
	if len(content) != 2 {
		t.Fatalf("expected reasoning + tool_use to be preserved, got %d items: %#v", len(content), content)
	}
	if content[0].Type != agent.ContentTypeRedactedThinking {
		t.Fatalf("expected first item to preserve reasoning as redacted_thinking passthrough, got %q", content[0].Type)
	}
	if content[1].Type != agent.ContentTypeToolUse || content[1].ToolUse == nil {
		t.Fatalf("expected second item to be tool_use, got %#v", content[1])
	}

	input := convertToResponsesInput([]agent.Message{{
		Role:    agent.RoleAssistant,
		Content: content,
	}})
	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	var items []map[string]any
	if err := json.Unmarshal(raw, &items); err != nil {
		t.Fatalf("json.Unmarshal replay items: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("expected replay to include reasoning + function_call items, got %d (%s)", len(items), string(raw))
	}
	if got := items[0]["type"]; got != "reasoning" {
		t.Fatalf("expected first replay item type=reasoning, got %v (%s)", got, string(raw))
	}
	if got := items[1]["type"]; got != "function_call" {
		t.Fatalf("expected second replay item type=function_call, got %v (%s)", got, string(raw))
	}
	if got := items[0]["id"]; got != "rs_123" {
		t.Fatalf("expected reasoning item id to round-trip, got %v (%s)", got, string(raw))
	}
}

func TestAgentLoopPreservesResponsesReasoningItemsAcrossToolIteration(t *testing.T) {
	t.Helper()

	var client loopClient
	client.t = t

	ag, err := agent.New(agent.Opts{
		SystemPrompt: "test",
		Client:       &client,
		ToolProvider: func() []*agent.Tool {
			return []*agent.Tool{
				agent.NewTool("get_weather", "Get weather", func(ctx context.Context, args struct {
					Location string `json:"location"`
				}) (string, []agent.MessageContent, error) {
					return "sunny", nil, nil
				}),
			}
		},
		Logf: func(string, ...any) {},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}

	resp, err := ag.Loop(context.Background(), []agent.Message{{
		Role: agent.RoleUser,
		Content: agent.Content{
			agent.NewTextContent("What's the weather in Dublin?"),
		},
	}}, nil)
	if err != nil {
		t.Fatalf("Agent.Loop: %v", err)
	}
	if resp == nil {
		t.Fatalf("expected final response")
	}
	if got := resp.Content.LossyText(); got != "Dublin is sunny." {
		t.Fatalf("final response mismatch: got %q", got)
	}
	if client.calls != 2 {
		t.Fatalf("expected exactly 2 LLM calls, got %d", client.calls)
	}
}

func (m *loopClient) CreateMessages(ctx context.Context, req agent.MessagesRequest, onUpdate func(agent.MessagesResponse)) (agent.MessagesResponse, error) {
	m.calls++
	switch m.calls {
	case 1:
		if len(req.Messages) != 1 {
			m.t.Fatalf("first call expected 1 message, got %d", len(req.Messages))
		}
		var output []responses.ResponseOutputItemUnion
		if err := json.Unmarshal([]byte(`[
			{
				"id":"rs_loop_1",
				"type":"reasoning",
				"summary":[{"type":"summary_text","text":"Need weather lookup first."}],
				"status":"completed"
			},
			{
				"id":"fc_loop_1",
				"type":"function_call",
				"call_id":"call_weather_1",
				"name":"get_weather",
				"arguments":"{\"location\":\"Dublin\"}",
				"status":"completed"
			}
		]`), &output); err != nil {
			m.t.Fatalf("json.Unmarshal first output: %v", err)
		}
		return agent.MessagesResponse{
			Content:    responseOutputToContent(output),
			StopReason: "tool_use",
		}, nil
	case 2:
		if len(req.Messages) < 3 {
			m.t.Fatalf("second call expected at least user+assistant+tool messages, got %d", len(req.Messages))
		}
		raw, err := json.Marshal(convertToResponsesInput(req.Messages))
		if err != nil {
			m.t.Fatalf("json.Marshal replay input: %v", err)
		}
		var items []map[string]any
		if err := json.Unmarshal(raw, &items); err != nil {
			m.t.Fatalf("json.Unmarshal replay input: %v", err)
		}
		var sawReasoning, sawFunctionCallOutput bool
		for _, it := range items {
			typ, _ := it["type"].(string)
			switch typ {
			case "reasoning":
				sawReasoning = true
			case "function_call_output":
				sawFunctionCallOutput = true
			}
		}
		if !sawReasoning {
			m.t.Fatalf("expected replay input to include reasoning item, got %s", string(raw))
		}
		if !sawFunctionCallOutput {
			m.t.Fatalf("expected replay input to include function_call_output, got %s", string(raw))
		}
		return agent.MessagesResponse{
			Content: agent.Content{
				agent.NewTextContent("Dublin is sunny."),
			},
			StopReason: "end_turn",
		}, nil
	default:
		m.t.Fatalf("unexpected LLM call %d", m.calls)
		return agent.MessagesResponse{}, nil
	}
}
