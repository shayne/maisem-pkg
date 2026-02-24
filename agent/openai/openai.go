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
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
	"pkg.maisem.dev/agent"
)

// Client wraps the OpenAI client to implement agent.LLMClient
type Client struct {
	client *openai.Client
	model  string
	logf   func(string, ...any)
	// retryEmptyCompletedMessageLimit controls how many automatic retries are
	// allowed for top-level
	// Responses calls that return a completed assistant message with only empty
	// output_text parts.
	retryEmptyCompletedMessageLimit int

	// responsesNewHook is used in tests to stub Responses API calls.
	responsesNewHook func(context.Context, responses.ResponseNewParams) (*responses.Response, error)
}

var _ agent.LLMClient = (*Client)(nil)
var _ agent.TokenCounter = (*Client)(nil)
var _ agent.PreviousResponseIDSupport = (*Client)(nil)

// ErrResponsesEmptyCompletedMessage is returned when the Responses API reports a
// completed assistant message but provides only empty output_text content.
var ErrResponsesEmptyCompletedMessage = errors.New("openai responses returned empty completed message")

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

// SetLogf installs an optional logger used for debug/observability messages.
func (c *Client) SetLogf(logf func(string, ...any)) {
	c.logf = logf
}

// SetRetryEmptyCompletedMessageOnce controls whether the client retries once
// when the Responses API returns a completed assistant message with no
// actionable content and only empty output_text parts on a top-level turn.
// Deprecated: prefer SetRetryEmptyCompletedMessageLimit.
func (c *Client) SetRetryEmptyCompletedMessageOnce(v bool) {
	if v {
		c.retryEmptyCompletedMessageLimit = 1
		return
	}
	c.retryEmptyCompletedMessageLimit = 0
}

// SetRetryEmptyCompletedMessageLimit controls how many times the client retries
// a top-level Responses call when it returns a completed assistant message with
// no actionable content and only empty output_text parts.
func (c *Client) SetRetryEmptyCompletedMessageLimit(limit int) {
	if limit < 0 {
		limit = 0
	}
	c.retryEmptyCompletedMessageLimit = limit
}

// SupportsPreviousResponseID reports whether this model is using the Responses API
// and can continue a conversation via previous_response_id.
func (c *Client) SupportsPreviousResponseID() bool {
	return usesResponsesAPI(c.model)
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

	if usesResponsesAPI(c.model) {
		return c.createMessagesWithResponses(ctx, req)
	}

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

func (c *Client) createMessagesWithResponses(ctx context.Context, req agent.MessagesRequest) (agent.MessagesResponse, error) {
	params := c.buildResponsesParams(req)
	prevResponseSet := req.ConversationState != nil && strings.TrimSpace(req.ConversationState.PreviousResponseID) != ""
	if c.logf != nil {
		c.logf(
			"openai_responses: request_state_mode=%s messages=%d prev_response_set=%t",
			responsesRequestStateMode(req),
			len(req.Messages),
			prevResponseSet,
		)
	}

	var (
		resp                    *responses.Response
		err                     error
		responseTime            time.Duration
		collectedContent        []agent.MessageContent
		droppedOutputTypes      map[string]int
		ignoredMessagePartTypes map[string]int
		emptyOutputRetryCount   int
	)
	for {
		start := time.Now()
		resp, err = c.responsesNew(ctx, params)
		if err != nil {
			return agent.MessagesResponse{}, c.handleError(err)
		}
		responseTime = time.Since(start)

		if resp.Status == "incomplete" && resp.IncompleteDetails.Reason == "max_output_tokens" {
			return agent.MessagesResponse{}, fmt.Errorf("%w: %s", agent.ErrTooLarge, resp.IncompleteDetails.Reason)
		}

		collectedContent, droppedOutputTypes, ignoredMessagePartTypes = responseOutputToContentWithUnhandled(resp.Output)
		if c.logf != nil && len(droppedOutputTypes) > 0 {
			c.logf("openai_responses: dropped_output_items=%s", formatTypeCountMap(droppedOutputTypes))
		}
		if c.logf != nil && len(ignoredMessagePartTypes) > 0 {
			c.logf("openai_responses: ignored_message_parts=%s", formatTypeCountMap(ignoredMessagePartTypes))
		}
		if !hasActionableResponsesContent(collectedContent) {
			diagnostics := responsesNoContentDiagnosticsSummary(resp, droppedOutputTypes, ignoredMessagePartTypes)
			if c.logf != nil {
				c.logf("openai_responses: no_actionable_content_details=%s", diagnostics)
			}
			if shouldRetryTopLevelEmptyOutputTextOnlyNoActionable(c.retryEmptyCompletedMessageLimit, prevResponseSet, req, resp, droppedOutputTypes, ignoredMessagePartTypes, emptyOutputRetryCount) {
				emptyOutputRetryCount++
				if c.logf != nil {
					c.logf("openai_responses: retry_triggered reason=top_level_empty_output_text_only attempt=%d", emptyOutputRetryCount)
				}
				continue
			}
			if err := unsupportedResponsesNoContentError(resp, droppedOutputTypes, ignoredMessagePartTypes, prevResponseSet); err != nil {
				if c.logf != nil {
					if emptyOutputRetryCount > 0 && errors.Is(err, ErrResponsesEmptyCompletedMessage) {
						c.logf("openai_responses: retry_outcome=failed reason=top_level_empty_output_text_only")
					}
					c.logf("openai_responses: %v", err)
				}
				return agent.MessagesResponse{}, err
			}
			if c.logf != nil && prevResponseSet && isBenignEmptyOutputTextOnlyNoop(resp.Output, droppedOutputTypes, ignoredMessagePartTypes) {
				c.logf("openai_responses: no_actionable_content treated_as=noop reason=output_text_empty_only")
			}
		}
		if emptyOutputRetryCount > 0 && c.logf != nil {
			c.logf("openai_responses: retry_outcome=success reason=top_level_empty_output_text_only")
		}
		break
	}
	responseUsage := agent.Usage{
		InputTokens:  int(resp.Usage.InputTokens),
		OutputTokens: int(resp.Usage.OutputTokens),
	}
	if resp.Usage.InputTokensDetails.CachedTokens != 0 {
		responseUsage.CacheReadInputTokens = int(resp.Usage.InputTokensDetails.CachedTokens)
	}

	if responseUsage.InputTokens == 0 && responseUsage.OutputTokens == 0 {
		if inputTokens, countErr := c.CountTokens(ctx, req); countErr == nil {
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
		responseUsage.OutputTokens = outputChars / 4
	}

	stopReason := "end_turn"
	for _, content := range collectedContent {
		if content.Type == agent.ContentTypeToolUse {
			stopReason = "tool_use"
			break
		}
	}

	return agent.MessagesResponse{
		ID:               resp.ID,
		StopReason:       stopReason,
		Content:          collectedContent,
		TimeToFirstToken: responseTime, // Responses API non-streaming call does not expose TTFT.
		ResponseTime:     responseTime,
		Usage:            responseUsage,
	}, nil
}

func (c *Client) responsesNew(ctx context.Context, params responses.ResponseNewParams) (*responses.Response, error) {
	if c.responsesNewHook != nil {
		return c.responsesNewHook(ctx, params)
	}
	return c.client.Responses.New(ctx, params)
}

func (c *Client) buildResponsesParams(req agent.MessagesRequest) responses.ResponseNewParams {
	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(c.model),
	}

	if req.System != "" {
		params.Instructions = openai.String(req.System)
	}
	if req.MaxTokens > 0 {
		params.MaxOutputTokens = openai.Int(int64(req.MaxTokens))
	}
	if req.ConversationState != nil {
		previousResponseID := strings.TrimSpace(req.ConversationState.PreviousResponseID)
		if previousResponseID != "" {
			params.PreviousResponseID = openai.String(previousResponseID)
		}
	}

	input := convertToResponsesInput(req.Messages)
	if len(input) > 0 {
		params.Input = responses.ResponseNewParamsInputUnion{
			OfInputItemList: input,
		}
	}

	if len(req.Tools) > 0 {
		params.Tools = make([]responses.ToolUnionParam, 0, len(req.Tools))
		for _, t := range req.Tools {
			var schema map[string]any
			if err := json.Unmarshal(t.InputSchema, &schema); err != nil {
				continue
			}
			params.Tools = append(params.Tools, responses.ToolUnionParam{
				OfFunction: &responses.FunctionToolParam{
					Name:        t.Name,
					Description: openai.String(t.Description),
					Parameters:  schema,
					// Responses strict mode requires all properties to be listed in required.
					// Keep tool schemas non-strict so optional fields remain valid.
					Strict: openai.Bool(false),
				},
			})
		}
	}

	if req.ToolChoice != nil {
		switch req.ToolChoice.Type {
		case "tool":
			params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
				OfFunctionTool: &responses.ToolChoiceFunctionParam{
					Name: req.ToolChoice.Name,
				},
			}
		case "any":
			params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
				OfToolChoiceMode: openai.Opt(responses.ToolChoiceOptionsRequired),
			}
		default:
			params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
				OfToolChoiceMode: openai.Opt(responses.ToolChoiceOptionsAuto),
			}
		}
	}

	return params
}

func convertToResponsesInput(messages []agent.Message) responses.ResponseInputParam {
	var out responses.ResponseInputParam
	nextAssistantMessageID := 1

	for _, m := range messages {
		switch m.Role {
		case agent.RoleTool:
			for _, c := range m.Content {
				if c.Type != agent.ContentTypeToolResult || c.ToolResults == nil {
					continue
				}
				out = append(out, responses.ResponseInputItemParamOfFunctionCallOutput(
					c.ToolResults.ToolCallID,
					c.ToolResults.Output,
				))
			}
		case agent.RoleAssistant:
			var assistantItems []responses.ResponseInputItemUnionParam
			assistantItems, nextAssistantMessageID = assistantMessageToResponsesItems(m.Content, nextAssistantMessageID)
			out = append(out, assistantItems...)
		case agent.RoleSystem:
			parts := contentToResponsesParts(m.Content)
			if len(parts) == 0 {
				continue
			}
			out = append(out, responses.ResponseInputItemParamOfMessage(parts, responses.EasyInputMessageRoleSystem))
		default:
			parts := contentToResponsesParts(m.Content)
			if len(parts) == 0 {
				continue
			}
			out = append(out, responses.ResponseInputItemParamOfMessage(parts, responses.EasyInputMessageRoleUser))
		}
	}

	return out
}

func assistantMessageToResponsesItems(content agent.Content, nextMessageID int) ([]responses.ResponseInputItemUnionParam, int) {
	var out []responses.ResponseInputItemUnionParam
	var textParts []responses.ResponseOutputMessageContentUnionParam

	flushText := func() {
		if len(textParts) == 0 {
			return
		}
		msgID := fmt.Sprintf("msg_history_assistant_%d", nextMessageID)
		nextMessageID++
		out = append(out, responses.ResponseInputItemParamOfOutputMessage(
			textParts,
			msgID,
			responses.ResponseOutputMessageStatusCompleted,
		))
		textParts = nil
	}

	for _, c := range content {
		switch c.Type {
		case agent.ContentTypeText:
			textParts = append(textParts, responses.ResponseOutputMessageContentUnionParam{
				OfOutputText: &responses.ResponseOutputTextParam{
					Text: c.Text,
				},
			})
		case agent.ContentTypeRedactedThinking:
			if reasoningItem, ok := decodeResponsesReasoningReplayItem(c.RedactedThinking); ok {
				flushText()
				out = append(out, reasoningItem)
			}
		case agent.ContentTypeToolUse:
			if c.ToolUse == nil {
				continue
			}
			flushText()
			callID := c.ToolUse.ID
			if callID == "" {
				callID = fmt.Sprintf("call_%d", len(out)+1)
			}
			out = append(out, responses.ResponseInputItemParamOfFunctionCall(
				string(normalizeToolCallArguments(string(c.ToolUse.Input))),
				callID,
				c.ToolUse.Name,
			))
		}
	}

	flushText()
	return out, nextMessageID
}

func contentToResponsesParts(content agent.Content) responses.ResponseInputMessageContentListParam {
	var parts responses.ResponseInputMessageContentListParam
	for _, c := range content {
		switch c.Type {
		case agent.ContentTypeText:
			parts = append(parts, responses.ResponseInputContentParamOfInputText(c.Text))
		case agent.ContentTypeImage:
			parts = append(parts, responses.ResponseInputContentUnionParam{
				OfInputImage: &responses.ResponseInputImageParam{
					Detail:   responses.ResponseInputImageDetailAuto,
					ImageURL: openai.String(fmt.Sprintf("data:%s;base64,%s", c.MediaType, c.Data)),
				},
			})
		}
	}
	return parts
}

func responseOutputToContent(output []responses.ResponseOutputItemUnion) []agent.MessageContent {
	content, _, _ := responseOutputToContentWithUnhandled(output)
	return content
}

func responseOutputToContentWithUnhandled(output []responses.ResponseOutputItemUnion) ([]agent.MessageContent, map[string]int, map[string]int) {
	var out []agent.MessageContent
	var dropped map[string]int
	var ignoredMessageParts map[string]int
	for _, item := range output {
		switch item.Type {
		case "message":
			msg := item.AsMessage()
			for _, part := range msg.Content {
				if part.Type == "output_text" && part.Text != "" {
					out = append(out, agent.NewTextContent(part.Text))
					continue
				}
				if part.Type == "refusal" && strings.TrimSpace(part.Refusal) != "" {
					out = append(out, agent.NewTextContent(part.Refusal))
					continue
				}
				key := strings.TrimSpace(part.Type)
				switch {
				case key == "":
					key = "unknown"
				case key == "output_text" && strings.TrimSpace(part.Text) == "":
					key = "output_text_empty"
				case key == "refusal" && strings.TrimSpace(part.Refusal) == "":
					key = "refusal_empty"
				}
				if ignoredMessageParts == nil {
					ignoredMessageParts = map[string]int{}
				}
				ignoredMessageParts[key]++
			}
		case "function_call":
			call := item.AsFunctionCall()
			callID := call.CallID
			if callID == "" {
				callID = call.ID
			}
			out = append(out, agent.MessageContent{
				Type: agent.ContentTypeToolUse,
				ToolUse: &agent.ToolUse{
					ID:    callID,
					Name:  call.Name,
					Input: normalizeToolCallArguments(call.Arguments),
				},
			})
		case "reasoning":
			reasoning := item.AsReasoning()
			if raw := strings.TrimSpace(reasoning.RawJSON()); raw != "" {
				out = append(out, agent.MessageContent{
					Type:             agent.ContentTypeRedactedThinking,
					RedactedThinking: encodeResponsesReasoningReplayItem(raw),
				})
			}
		default:
			if item.Type != "" {
				if dropped == nil {
					dropped = map[string]int{}
				}
				dropped[item.Type]++
			}
		}
	}
	return out, dropped, ignoredMessageParts
}

func hasActionableResponsesContent(content []agent.MessageContent) bool {
	for _, c := range content {
		switch c.Type {
		case agent.ContentTypeText:
			if strings.TrimSpace(c.Text) != "" {
				return true
			}
		case agent.ContentTypeToolUse:
			if c.ToolUse != nil {
				return true
			}
		}
	}
	return false
}

func unsupportedResponsesNoContentError(resp *responses.Response, droppedOutputTypes, ignoredMessagePartTypes map[string]int, allowBenignEmptyOutputTextOnlyNoop bool) error {
	if resp == nil {
		return nil
	}
	if len(resp.Output) == 0 {
		return nil
	}
	if isBenignEmptyOutputTextOnlyNoop(resp.Output, droppedOutputTypes, ignoredMessagePartTypes) {
		if allowBenignEmptyOutputTextOnlyNoop {
			return nil
		}
		return fmt.Errorf("%w (%s)", ErrResponsesEmptyCompletedMessage, responsesNoContentDiagnosticsSummary(resp, droppedOutputTypes, ignoredMessagePartTypes))
	}
	return fmt.Errorf("openai responses returned no supported actionable content (%s)", responsesNoContentDiagnosticsSummary(resp, droppedOutputTypes, ignoredMessagePartTypes))
}

func responsesNoContentDiagnosticsSummary(resp *responses.Response, droppedOutputTypes, ignoredMessagePartTypes map[string]int) string {
	if resp == nil {
		return "response=nil"
	}
	outputItemTypes := make(map[string]int)
	for _, item := range resp.Output {
		typ := strings.TrimSpace(item.Type)
		if typ == "" {
			typ = "unknown"
		}
		outputItemTypes[typ]++
	}
	responseStatus := strings.TrimSpace(string(resp.Status))
	if responseStatus == "" {
		responseStatus = "unknown"
	}
	var details []string
	details = append(details, fmt.Sprintf("response_status=%s", responseStatus))
	if reason := strings.TrimSpace(resp.IncompleteDetails.Reason); reason != "" {
		details = append(details, fmt.Sprintf("incomplete_reason=%s", reason))
	}
	if code := strings.TrimSpace(string(resp.Error.Code)); code != "" {
		details = append(details, fmt.Sprintf("response_error_code=%s", code))
	}
	if msg := strings.TrimSpace(resp.Error.Message); msg != "" {
		details = append(details, "response_error_message_present=true")
	}
	details = append(details, fmt.Sprintf("output_items=%s", formatTypeCountMap(outputItemTypes)))
	if itemStatuses := responseOutputItemStatusCounts(resp.Output); len(itemStatuses) > 0 {
		details = append(details, fmt.Sprintf("output_item_statuses=%s", formatTypeCountMap(itemStatuses)))
	}
	if len(ignoredMessagePartTypes) > 0 {
		details = append(details, fmt.Sprintf("ignored_message_parts=%s", formatTypeCountMap(ignoredMessagePartTypes)))
	}
	if len(droppedOutputTypes) > 0 {
		details = append(details, fmt.Sprintf("dropped_output_items=%s", formatTypeCountMap(droppedOutputTypes)))
	}
	return strings.Join(details, " ")
}

func responseOutputItemStatusCounts(output []responses.ResponseOutputItemUnion) map[string]int {
	var counts map[string]int
	for _, item := range output {
		type itemMeta struct {
			Type   string `json:"type"`
			Status string `json:"status"`
		}
		typ := strings.TrimSpace(item.Type)
		status := ""
		if raw := strings.TrimSpace(item.RawJSON()); raw != "" {
			var meta itemMeta
			if err := json.Unmarshal([]byte(raw), &meta); err == nil {
				if strings.TrimSpace(meta.Type) != "" {
					typ = strings.TrimSpace(meta.Type)
				}
				status = strings.TrimSpace(meta.Status)
			}
		}
		if typ == "" {
			typ = "unknown"
		}
		if status == "" {
			status = "none"
		}
		if counts == nil {
			counts = map[string]int{}
		}
		counts[fmt.Sprintf("%s:%s", typ, status)]++
	}
	return counts
}

func shouldRetryTopLevelEmptyOutputTextOnlyNoActionable(retryLimit int, prevResponseSet bool, req agent.MessagesRequest, resp *responses.Response, droppedOutputTypes, ignoredMessagePartTypes map[string]int, retryCount int) bool {
	if retryLimit <= 0 || retryCount >= retryLimit || prevResponseSet || resp == nil {
		return false
	}
	if len(req.Messages) == 0 {
		return false
	}
	return isBenignEmptyOutputTextOnlyNoop(resp.Output, droppedOutputTypes, ignoredMessagePartTypes)
}

func isBenignEmptyOutputTextOnlyNoop(output []responses.ResponseOutputItemUnion, droppedOutputTypes, ignoredMessagePartTypes map[string]int) bool {
	if len(output) == 0 || len(droppedOutputTypes) > 0 || len(ignoredMessagePartTypes) == 0 {
		return false
	}
	for typ, count := range ignoredMessagePartTypes {
		if typ != "output_text_empty" || count <= 0 {
			return false
		}
	}
	for _, item := range output {
		if strings.TrimSpace(item.Type) != "message" {
			return false
		}
	}
	return true
}

func formatTypeCountMap(m map[string]int) string {
	if len(m) == 0 {
		return "none"
	}
	parts := make([]string, 0, len(m))
	for typ, count := range m {
		parts = append(parts, fmt.Sprintf("%s=%d", typ, count))
	}
	sort.Strings(parts)
	return strings.Join(parts, ",")
}

func responsesRequestStateMode(req agent.MessagesRequest) string {
	prevSet := req.ConversationState != nil && strings.TrimSpace(req.ConversationState.PreviousResponseID) != ""
	switch {
	case prevSet:
		return "previous_response_id_continuation"
	case len(req.Messages) > 0:
		return "manual_replay"
	default:
		return "stateless_empty"
	}
}

const responsesReasoningReplayPrefix = "openai_responses_reasoning:"

func encodeResponsesReasoningReplayItem(raw string) string {
	return responsesReasoningReplayPrefix + raw
}

func decodeResponsesReasoningReplayItem(v string) (responses.ResponseInputItemUnionParam, bool) {
	if !strings.HasPrefix(v, responsesReasoningReplayPrefix) {
		return responses.ResponseInputItemUnionParam{}, false
	}
	raw := strings.TrimSpace(strings.TrimPrefix(v, responsesReasoningReplayPrefix))
	if raw == "" {
		return responses.ResponseInputItemUnionParam{}, false
	}
	var reasoning responses.ResponseReasoningItem
	if err := json.Unmarshal([]byte(raw), &reasoning); err != nil {
		return responses.ResponseInputItemUnionParam{}, false
	}
	p := reasoning.ToParam()
	return responses.ResponseInputItemUnionParam{OfReasoning: &p}, true
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

func usesResponsesAPI(model string) bool {
	m := strings.ToLower(strings.TrimSpace(model))
	return strings.Contains(m, "codex")
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
