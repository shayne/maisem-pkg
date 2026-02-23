// Copyright (c) 2025 AUTHORS All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
	"pkg.maisem.dev/agent"
)

// Client wraps the OpenAI client to implement agent.LLMClient
type Client struct {
	client *openai.Client
	model  string
}

var _ agent.LLMClient = (*Client)(nil)
var _ agent.TokenCounter = (*Client)(nil)

// New creates a new Client
func New(apiKey, model string) *Client {
	client := openai.NewClient(option.WithAPIKey(apiKey))
	return &Client{client: &client, model: model}
}

// NewWithOptions creates a new Client with custom options
func NewWithOptions(model string, opts ...option.RequestOption) *Client {
	client := openai.NewClient(opts...)
	return &Client{client: &client, model: model}
}

// SetModel sets the model for the client
func (c *Client) SetModel(model string) {
	c.model = model
}

// CountTokens estimates the number of tokens in a request.
// OpenAI doesn't provide a direct token counting API, so this is a rough estimation.
// A more accurate implementation would use tiktoken or similar tokenizer.
func (c *Client) CountTokens(ctx context.Context, req agent.MessagesRequest) (int, error) {
	// Rough estimation: ~4 characters per token
	// This is a simplified approach. For production use, consider using tiktoken.
	totalChars := 0

	// Count system prompt
	totalChars += len(req.System)

	// Count messages
	for _, msg := range req.Messages {
		for _, content := range msg.Content {
			switch content.Type {
			case agent.ContentTypeText:
				totalChars += len(content.Text)
			case agent.ContentTypeToolUse:
				if content.ToolUse != nil {
					totalChars += len(content.ToolUse.Name) + len(content.ToolUse.Input)
				}
			case agent.ContentTypeToolResult:
				if content.ToolResults != nil {
					totalChars += len(content.ToolResults.Output)
				}
			}
		}
	}

	// Count tool definitions
	for _, tool := range req.Tools {
		totalChars += len(tool.Name) + len(tool.Description) + len(tool.InputSchema)
	}

	// Rough estimate: 4 characters per token
	estimatedTokens := totalChars / 4

	return estimatedTokens, nil
}

func (c *Client) CreateMessages(ctx context.Context, req agent.MessagesRequest, onUpdate func(agent.MessagesResponse)) (agent.MessagesResponse, error) {
	_ = onUpdate

	// Convert our request to OpenAI request
	params := c.buildChatCompletionParams(req)

	// Make the request
	start := time.Now()
	timeToFirstToken := time.Duration(0)
	stream := c.client.Chat.Completions.NewStreaming(ctx, params)

	var collectedContent []agent.MessageContent
	toolCallsByIndex := map[int64]*streamedToolCall{}
	var toolCallOrder []int64
	var id string
	var finishReason string
	var usage openai.CompletionUsage

	for stream.Next() {
		chunk := stream.Current()

		// Set TTFT on first chunk
		if timeToFirstToken == 0 {
			timeToFirstToken = time.Since(start)
		}

		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]
			id = chunk.ID

			// Handle text content
			if choice.Delta.Content != "" {
				var textContent *agent.MessageContent
				for i := range collectedContent {
					if collectedContent[i].Type == agent.ContentTypeText {
						textContent = &collectedContent[i]
						break
					}
				}
				if textContent == nil {
					collectedContent = append(collectedContent, agent.MessageContent{
						Type: agent.ContentTypeText,
						Text: "",
					})
					textContent = &collectedContent[len(collectedContent)-1]
				}
				textContent.Text += choice.Delta.Content
			}

			// Handle tool calls (arguments stream across chunks).
			collectStreamToolCalls(choice.Delta.ToolCalls, toolCallsByIndex, &toolCallOrder)

			// Set finish reason
			if choice.FinishReason != "" {
				finishReason = choice.FinishReason
			}
		}

		// Capture usage from any chunk that has it
		if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
			usage = chunk.Usage
		}
	}

	if err := stream.Err(); err != nil {
		return agent.MessagesResponse{}, c.handleError(err)
	}

	responseTime := time.Since(start)

	// Handle max tokens case with error
	if finishReason == "length" {
		return agent.MessagesResponse{}, fmt.Errorf("%w: %v", agent.ErrTooLarge, finishReason)
	}

	// Finalize streamed tool calls into complete tool_use content blocks.
	collectedContent = append(collectedContent, buildToolUseContents(toolCallsByIndex, toolCallOrder)...)

	// Build response with usage data
	responseUsage := agent.Usage{
		InputTokens:  int(usage.PromptTokens),
		OutputTokens: int(usage.CompletionTokens),
	}

	// Use token counting estimation if counts were not provided
	if usage.PromptTokens == 0 && usage.CompletionTokens == 0 {
		if inputTokens, err := c.CountTokens(ctx, req); err == nil {
			responseUsage.InputTokens = inputTokens
		}
		outputChars := 0
		for _, content := range collectedContent {
			if content.Type == agent.ContentTypeText {
				outputChars += len(content.Text)
			} else if content.Type == agent.ContentTypeToolUse && content.ToolUse != nil {
				outputChars += len(content.ToolUse.Name) + len(content.ToolUse.Input)
			}
		}
		responseUsage.OutputTokens = outputChars / 4 // Rough estimate: ~4 chars per token
	}

	response := agent.MessagesResponse{
		ID:               id,
		StopReason:       finishReason,
		Content:          collectedContent,
		TimeToFirstToken: timeToFirstToken,
		ResponseTime:     responseTime,
		Usage:            responseUsage,
	}

	// Add cache tokens if available
	if usage.PromptTokensDetails.CachedTokens != 0 {
		response.Usage.CacheReadInputTokens = int(usage.PromptTokensDetails.CachedTokens)
	}

	return response, nil
}

type streamedToolCall struct {
	ID        string
	Name      string
	Arguments strings.Builder
}

func collectStreamToolCalls(deltas []openai.ChatCompletionChunkChoiceDeltaToolCall, byIndex map[int64]*streamedToolCall, order *[]int64) {
	for _, tc := range deltas {
		state, ok := byIndex[tc.Index]
		if !ok {
			state = &streamedToolCall{}
			byIndex[tc.Index] = state
			*order = append(*order, tc.Index)
		}
		if tc.ID != "" {
			state.ID = tc.ID
		}
		if tc.Function.Name != "" {
			state.Name = tc.Function.Name
		}
		if tc.Function.Arguments != "" {
			state.Arguments.WriteString(tc.Function.Arguments)
		}
	}
}

func buildToolUseContents(byIndex map[int64]*streamedToolCall, order []int64) []agent.MessageContent {
	if len(order) == 0 {
		return nil
	}

	ordered := append([]int64(nil), order...)
	sort.Slice(ordered, func(i, j int) bool {
		return ordered[i] < ordered[j]
	})

	out := make([]agent.MessageContent, 0, len(ordered))
	for _, idx := range ordered {
		tc := byIndex[idx]
		if tc == nil || tc.Name == "" {
			continue
		}

		id := tc.ID
		if id == "" {
			id = fmt.Sprintf("call_%d", idx)
		}

		out = append(out, agent.MessageContent{
			Type: agent.ContentTypeToolUse,
			ToolUse: &agent.ToolUse{
				ID:    id,
				Name:  tc.Name,
				Input: normalizeToolCallArguments(tc.Arguments.String()),
			},
		})
	}

	return out
}

func normalizeToolCallArguments(raw string) json.RawMessage {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return json.RawMessage("{}")
	}
	if json.Valid([]byte(trimmed)) {
		return json.RawMessage(trimmed)
	}

	// Keep invalid model output as a JSON string literal so persistence succeeds
	// and downstream tool decoding can fail safely and explicitly.
	fallback, err := json.Marshal(trimmed)
	if err != nil {
		return json.RawMessage(`""`)
	}
	return json.RawMessage(fallback)
}

// buildChatCompletionParams converts agent request to OpenAI params
func (c *Client) buildChatCompletionParams(req agent.MessagesRequest) openai.ChatCompletionNewParams {
	params := openai.ChatCompletionNewParams{
		Model: c.model, // Model is a string type alias
	}

	// Set max tokens if specified
	if req.MaxTokens > 0 {
		if usesMaxCompletionTokens(c.model) {
			// Parameter changed in OpenAI API for GPT-5 and reasoning models.
			params.MaxCompletionTokens = openai.Int(int64(req.MaxTokens))
		} else {
			params.MaxTokens = openai.Int(int64(req.MaxTokens))
		}
	}

	// Add system message if present
	if req.System != "" {
		params.Messages = append(params.Messages, openai.SystemMessage(req.System))
	}

	// Convert messages
	for _, m := range req.Messages {
		params.Messages = append(params.Messages, convertMessage(m)...)
	}

	// Convert tools
	if len(req.Tools) > 0 {
		for _, t := range req.Tools {
			var schema map[string]any
			if err := json.Unmarshal(t.InputSchema, &schema); err != nil {
				// Skip tool if schema is invalid
				continue
			}

			params.Tools = append(params.Tools, openai.ChatCompletionToolParam{
				Function: shared.FunctionDefinitionParam{
					Name:        t.Name,
					Description: openai.String(t.Description),
					Parameters:  shared.FunctionParameters(schema),
				},
			})
		}
	}

	// Set tool choice
	if req.ToolChoice != nil {
		switch req.ToolChoice.Type {
		case "tool":
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionParamOfChatCompletionNamedToolChoice(
				openai.ChatCompletionNamedToolChoiceFunctionParam{
					Name: req.ToolChoice.Name,
				},
			)
		case "any":
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String("required"),
			}
		default:
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String("auto"),
			}
		}
	}

	return params
}

func usesMaxCompletionTokens(model string) bool {
	m := strings.ToLower(strings.TrimSpace(model))
	return strings.HasPrefix(m, "gpt-5") || strings.HasPrefix(m, "o3")
}

// convertMessage converts agent message to OpenAI format
func convertMessage(m agent.Message) []openai.ChatCompletionMessageParamUnion {
	var messages []openai.ChatCompletionMessageParamUnion

	switch m.Role {
	case agent.RoleTool:
		// Tool responses need special handling in OpenAI
		for _, c := range m.Content {
			if c.Type == agent.ContentTypeToolResult {
				messages = append(messages, openai.ToolMessage(c.ToolResults.Output, c.ToolResults.ToolCallID))
			}
		}
		return messages

	case agent.RoleAssistant:
		// Assistant messages
		var textContent string
		var toolCalls []openai.ChatCompletionMessageToolCallParam

		for _, c := range m.Content {
			switch c.Type {
			case agent.ContentTypeText:
				textContent += c.Text
			case agent.ContentTypeToolUse:
				toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallParam{
					ID: c.ToolUse.ID,
					Function: openai.ChatCompletionMessageToolCallFunctionParam{
						Name:      c.ToolUse.Name,
						Arguments: string(c.ToolUse.Input),
					},
				})
			case agent.ContentTypeThinking, agent.ContentTypeRedactedThinking:
				// OpenAI doesn't support thinking blocks, convert to text
				if c.Type == agent.ContentTypeThinking {
					textContent += fmt.Sprintf("\n<thinking>\n%s\n</thinking>\n", c.Thinking)
				}
			}
		}

		if len(toolCalls) > 0 {
			// Create assistant message with tool calls
			assistantMsg := openai.ChatCompletionAssistantMessageParam{
				ToolCalls: toolCalls,
			}
			if textContent != "" {
				assistantMsg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String(textContent),
				}
			}
			messages = append(messages, openai.ChatCompletionMessageParamUnion{
				OfAssistant: &assistantMsg,
			})
		} else if textContent != "" {
			messages = append(messages, openai.AssistantMessage(textContent))
		}

	case agent.RoleUser, agent.RoleSystem:
		// User messages
		var parts []openai.ChatCompletionContentPartUnionParam
		hasMultiContent := false

		for _, c := range m.Content {
			switch c.Type {
			case agent.ContentTypeText:
				parts = append(parts, openai.TextContentPart(c.Text))
			case agent.ContentTypeImage:
				hasMultiContent = true
				parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
					URL: fmt.Sprintf("data:%s;base64,%s", c.MediaType, c.Data),
				}))
			}
		}

		if hasMultiContent || len(parts) > 1 {
			messages = append(messages, openai.UserMessage(parts))
		} else if len(parts) == 1 {
			// Single text content - extract the text
			if textPart := parts[0].GetText(); textPart != nil {
				messages = append(messages, openai.UserMessage(*textPart))
			}
		}
	}

	return messages
}

// handleError converts OpenAI errors to agent errors
func (c *Client) handleError(err error) error {
	if err == nil {
		return nil
	}
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		if apiErr.StatusCode == 400 {
			return fmt.Errorf("%w: %v", agent.ErrTooLarge, err)
		}
		if apiErr.StatusCode == 429 {
			return &agent.RateLimitError{RetryAfter: -1, Err: err}
		}
		if apiErr.StatusCode == 503 {
			return fmt.Errorf("%w: %v", agent.ErrOverloaded, err)
		}
		if apiErr.StatusCode >= 500 {
			return fmt.Errorf("%w: %v", agent.ErrRemoteServerError, err)
		}
	}

	// Check for context length errors
	if strings.Contains(err.Error(), "context_length_exceeded") ||
		strings.Contains(err.Error(), "maximum context length") ||
		strings.Contains(err.Error(), "too many tokens") {
		return fmt.Errorf("%w: %v", agent.ErrTooLarge, err)
	}

	return err
}
