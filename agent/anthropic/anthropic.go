// Copyright (c) 2025 AUTHORS All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/liushuangls/go-anthropic/v2"
	"pkg.maisem.dev/agent"
)

// Client wraps the anthropic client to implement agent.LLMClient
type Client struct {
	client *anthropic.Client
	model  string
}

var _ agent.LLMClient = (*Client)(nil)

// New creates a new Client
func New(client *anthropic.Client, model string) *Client {
	return &Client{client: client, model: model}
}

func (c *Client) CountTokens(ctx context.Context, req agent.MessagesRequest) (int, error) {
	anthropicReq := c.convertMessagesRequest(req)
	anthropicReq.MaxTokens = 0
	resp, err := c.client.CountTokens(ctx, anthropicReq)
	if err != nil {
		return 0, err
	}
	return resp.InputTokens, nil
}

func (c *Client) convertMessagesRequest(req agent.MessagesRequest) anthropic.MessagesRequest {
	// Convert our request to anthropic request
	anthropicReq := anthropic.MessagesRequest{
		Model: anthropic.Model(c.model),
		MultiSystem: []anthropic.MessageSystemPart{
			{
				Type: "text",
				Text: req.System,
				CacheControl: &anthropic.MessageCacheControl{
					Type: anthropic.CacheControlTypeEphemeral,
				},
			},
		},
		MaxTokens: req.MaxTokens,
	}

	// Set thinking mode if enabled
	if req.ThinkingMode {
		thinkingTokens := 1024 // Default thinking token budget

		// First check if request includes specific token amount
		if req.ThinkingTokens > 0 {
			thinkingTokens = req.ThinkingTokens
		}

		anthropicReq.Thinking = &anthropic.Thinking{
			Type:         anthropic.ThinkingTypeEnabled,
			BudgetTokens: thinkingTokens,
		}
	}
	if req.ToolChoice != nil {
		anthropicReq.ToolChoice = &anthropic.ToolChoice{
			Type: req.ToolChoice.Type,
			Name: req.ToolChoice.Name,
		}
	}

	// Convert messages
	anthropicReq.Messages = make([]anthropic.Message, 0, len(req.Messages))
	for _, m := range req.Messages {
		if len(m.Content) > 0 {
			anthropicReq.Messages = append(anthropicReq.Messages, toAnthropicMessage(m))
		}
	}

	// Convert tools
	anthropicReq.Tools = make([]anthropic.ToolDefinition, len(req.Tools))
	for i, t := range req.Tools {
		anthropicReq.Tools[i] = anthropic.ToolDefinition{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		}
	}
	return anthropicReq
}

func (c *Client) CreateMessages(ctx context.Context, req agent.MessagesRequest, onUpdate func(agent.MessagesResponse)) (agent.MessagesResponse, error) {
	anthropicReq := c.convertMessagesRequest(req)
	start := time.Now()
	timeToFirstToken := time.Duration(0)

	// Track accumulated content for streaming updates
	var accumulatedContent []agent.MessageContent
	var currentToolUse *agent.MessageContent
	messageID := ""

	resp, err := c.client.CreateMessagesStream(ctx, anthropic.MessagesStreamRequest{
		MessagesRequest: anthropicReq,

		OnMessageStart: func(data anthropic.MessagesEventMessageStartData) {
			messageID = data.Message.ID
		},

		OnContentBlockStart: func(data anthropic.MessagesEventContentBlockStartData) {
			if timeToFirstToken == 0 {
				timeToFirstToken = time.Since(start)
			}

			switch data.ContentBlock.Type {
			case "text":
				content := agent.MessageContent{
					Type: agent.ContentTypeText,
					Text: "",
				}
				accumulatedContent = append(accumulatedContent, content)
			case "tool_use":
				if data.ContentBlock.MessageContentToolUse != nil {
					content := agent.MessageContent{
						Type: agent.ContentTypeToolUse,
						ToolUse: &agent.ToolUse{
							ID:    data.ContentBlock.ID,
							Name:  data.ContentBlock.Name,
							Input: data.ContentBlock.Input,
						},
					}
					newIndex := len(accumulatedContent)
					accumulatedContent = append(accumulatedContent, content)
					currentToolUse = &accumulatedContent[newIndex]
				}
			}
		},

		OnContentBlockDelta: func(data anthropic.MessagesEventContentBlockDeltaData) {
			if timeToFirstToken == 0 {
				timeToFirstToken = time.Since(start)
			}

			if len(accumulatedContent) > 0 {
				lastIdx := len(accumulatedContent) - 1
				switch data.Delta.Type {
				case "text_delta":
					if data.Delta.Text != nil {
						accumulatedContent[lastIdx].Text += *data.Delta.Text
						if onUpdate != nil {
							onUpdate(agent.MessagesResponse{
								ID:               messageID,
								StopReason:       "", // Not finished yet
								Content:          agent.Content(accumulatedContent),
								TimeToFirstToken: timeToFirstToken,
								ResponseTime:     time.Since(start),
							})
						}
					}
				case "input_json_delta":
					if currentToolUse != nil && currentToolUse.ToolUse != nil {
						// Accumulate JSON fragments and wait for BlockStop to call
						// onUpdate to avoid needing to handle partial JSON down the line
						if data.Delta.PartialJson != nil {
							if currentToolUse.ToolUse.Input == nil {
								currentToolUse.ToolUse.Input = json.RawMessage(*data.Delta.PartialJson)
							} else {
								// Append to existing JSON
								existing := string(currentToolUse.ToolUse.Input)
								currentToolUse.ToolUse.Input = json.RawMessage(existing + *data.Delta.PartialJson)
							}
						}
					}
				}
			}
		},

		OnContentBlockStop: func(data anthropic.MessagesEventContentBlockStopData, content anthropic.MessageContent) {
			if onUpdate != nil {
				onUpdate(agent.MessagesResponse{
					ID:               messageID,
					StopReason:       "", // Not finished yet
					Content:          agent.Content(accumulatedContent),
					TimeToFirstToken: timeToFirstToken,
					ResponseTime:     time.Since(start),
				})
			}
			currentToolUse = nil
		},
	})
	responseTime := time.Since(start)
	if err != nil {
		var apiErr *anthropic.APIError
		var reqErr *anthropic.RequestError
		if errors.As(err, &apiErr) {
			switch apiErr.Type {
			case anthropic.ErrTypeRateLimit:
				rlh, err := resp.GetRateLimitHeaders()
				if err != nil {
					return agent.MessagesResponse{}, &agent.RateLimitError{RetryAfter: 5 * time.Second, Err: apiErr}
				}
				return agent.MessagesResponse{}, &agent.RateLimitError{RetryAfter: time.Duration(rlh.RetryAfter) * time.Second, Err: apiErr}
			case anthropic.ErrTypeTooLarge:
				return agent.MessagesResponse{}, fmt.Errorf("%w: %v", agent.ErrTooLarge, apiErr)
			case anthropic.ErrTypeOverloaded:
				return agent.MessagesResponse{}, fmt.Errorf("%w: %v", agent.ErrOverloaded, apiErr)
			case anthropic.ErrTypeInvalidRequest:
				if strings.Contains(apiErr.Message, "too long") {
					return agent.MessagesResponse{}, fmt.Errorf("%w: %v", agent.ErrTooLarge, apiErr)
				}
			default:
				if apiErr.Message == "Internal server error" {
					return agent.MessagesResponse{}, &agent.RateLimitError{RetryAfter: time.Second, Err: apiErr}
				}
			}
		}
		if errors.As(err, &reqErr) && reqErr.StatusCode >= 500 {
			return agent.MessagesResponse{}, &agent.RateLimitError{RetryAfter: time.Second, Err: reqErr}
		}
		return agent.MessagesResponse{}, err
	}

	// Convert response using fromAnthropicMessage
	msg := fromAnthropicMessage(anthropic.Message{
		Role:    resp.Role,
		Content: resp.Content,
	})
	switch resp.StopReason {
	case anthropic.MessagesStopReasonEndTurn, anthropic.MessagesStopReasonToolUse:
	case anthropic.MessagesStopReasonMaxTokens:
		return agent.MessagesResponse{}, &agent.RateLimitError{RetryAfter: 0, Err: fmt.Errorf("max tokens reached")}
	default:
		log.Printf("Stop reason: %s", resp.StopReason)
	}

	response := agent.MessagesResponse{
		ID:         resp.ID,
		StopReason: string(resp.StopReason),
		Content:    msg.Content,
		Usage: agent.Usage{
			InputTokens:              resp.Usage.InputTokens,
			OutputTokens:             resp.Usage.OutputTokens,
			CacheCreationInputTokens: resp.Usage.CacheCreationInputTokens,
			CacheReadInputTokens:     resp.Usage.CacheReadInputTokens,
		},
		TimeToFirstToken: timeToFirstToken,
		ResponseTime:     responseTime,
	}

	return response, nil
}

// Convert anthropic types to our types
func fromAnthropicMessage(m anthropic.Message) agent.Message {
	contents := make([]agent.MessageContent, len(m.Content))
	for i, c := range m.Content {
		content := agent.MessageContent{
			Type: agent.ContentType(c.Type),
		}
		switch c.Type {
		case anthropic.MessagesContentTypeThinking:
			content.Thinking = c.Thinking
			content.Signature = c.Signature
		case anthropic.MessagesContentTypeRedactedThinking:
			content.RedactedThinking = c.Data
		case anthropic.MessagesContentTypeText:
			content.Text = *c.Text
		case anthropic.MessagesContentTypeToolUse:
			content.ToolUse = &agent.ToolUse{
				ID:    c.MessageContentToolUse.ID,
				Name:  c.MessageContentToolUse.Name,
				Input: c.MessageContentToolUse.Input,
			}
			if len(c.MessageContentToolUse.Input) == 0 {
				content.ToolUse.Input = json.RawMessage("{}")
			}
		case anthropic.MessagesContentTypeImage:
			panic("image seen from anthropic")
		case anthropic.MessagesContentTypeDocument:
			panic("document seen from anthropic")
		case anthropic.MessagesContentTypeToolResult:
			panic("tool_result seen from anthropic")
		case anthropic.MessagesContentTypeInputJsonDelta,
			anthropic.MessagesContentTypeCitationsDelta,
			anthropic.MessagesContentTypeThinkingDelta,
			anthropic.MessagesContentTypeSignatureDelta:
			panic("delta seen from anthropic")
		}
		if c.CacheControl != nil {
			content.CacheControl = &agent.CacheControl{
				Type: string(c.CacheControl.Type),
			}
		}
		contents[i] = content
	}
	return agent.Message{
		Role:    agent.Role(m.Role),
		Content: contents,
	}
}

func toAnthropicMessage(m agent.Message) anthropic.Message {
	contents := make([]anthropic.MessageContent, 0, len(m.Content))
	for _, c := range m.Content {
		content := anthropic.MessageContent{
			Type: anthropic.MessagesContentType(c.Type),
		}
		switch c.Type {
		case agent.ContentTypeText:
			content.Text = &c.Text
		case agent.ContentTypeImage:
			content.Source = &anthropic.MessageContentSource{
				Data:      c.Data,
				MediaType: c.MediaType,
				Type:      "base64",
			}
		case agent.ContentTypeRedactedThinking:
			content.MessageContentRedactedThinking = &anthropic.MessageContentRedactedThinking{
				Data: c.RedactedThinking,
			}
		case agent.ContentTypeThinking:
			content.MessageContentThinking = &anthropic.MessageContentThinking{
				Thinking:  c.Thinking,
				Signature: c.Signature,
			}
		case agent.ContentTypeToolUse:
			content.MessageContentToolUse = anthropic.NewMessageContentToolUse(c.ToolUse.ID, c.ToolUse.Name, c.ToolUse.Input)
		case agent.ContentTypeToolResult:
			content.MessageContentToolResult = anthropic.NewMessageContentToolResult(c.ToolResults.ToolCallID, c.ToolResults.Output, c.ToolResults.Error)
		}
		if c.CacheControl != nil {
			content.CacheControl = &anthropic.MessageCacheControl{
				Type: anthropic.CacheControlType(c.CacheControl.Type),
			}
		}
		contents = append(contents, content)
	}
	role := anthropic.ChatRole(m.Role)
	if role == "tool" {
		role = anthropic.RoleUser
	}
	return anthropic.Message{
		Role:    role,
		Content: contents,
	}
}
