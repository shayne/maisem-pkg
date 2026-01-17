// Copyright (c) 2025 AUTHORS All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"google.golang.org/genai"
	"pkg.maisem.dev/agent"
)

// Client wraps the Gemini client to implement agent.LLMClient
type Client struct {
	client *genai.Client
	model  string
}

var _ agent.LLMClient = (*Client)(nil)

// New creates a new Client
func New(client *genai.Client, modelName string) *Client {
	return &Client{
		client: client,
		model:  modelName,
	}
}

func (c *Client) convertMessagesRequest(req agent.MessagesRequest) ([]*genai.Content, error) {
	var msgs []*genai.Content
	for _, msg := range req.Messages {
		var gc []*genai.Part
		for _, c := range msg.Content {
			switch c.Type {
			case agent.ContentTypeText:
				gc = append(gc, genai.NewPartFromText(c.Text))
			case agent.ContentTypeToolUse:
				var args map[string]any
				if err := json.Unmarshal(c.ToolUse.Input, &args); err != nil {
					return nil, err
				}
				gc = append(gc, &genai.Part{
					FunctionCall: &genai.FunctionCall{
						ID:   c.ToolUse.ID,
						Name: c.ToolUse.Name,
						Args: args,
					},
				})
			case agent.ContentTypeToolResult:
				response := map[string]any{}
				if c.ToolResults.Output != "" {
					response["output"] = c.ToolResults.Output
				}
				if c.ToolResults.Error {
					response["error"] = true
				}
				gc = append(gc, &genai.Part{
					FunctionResponse: &genai.FunctionResponse{
						ID:       c.ToolResults.ToolCallID,
						Name:     c.ToolResults.Name,
						Response: response,
					},
				})
			case agent.ContentTypeThinking, agent.ContentTypeRedactedThinking:
				return nil, errors.New("gemini does not support thinking or redacted thinking")
			}
		}
		switch msg.Role {
		case agent.RoleUser, agent.RoleTool:
			msgs = append(msgs, genai.NewContentFromParts(gc, "user"))
		case agent.RoleAssistant:
			msgs = append(msgs, &genai.Content{
				Role:  "model",
				Parts: gc,
			})
		}
	}
	return msgs, nil
}

func (c *Client) CreateMessages(ctx context.Context, req agent.MessagesRequest, onUpdate func(agent.MessagesResponse)) (agent.MessagesResponse, error) {
	msgs, err := c.convertMessagesRequest(req)
	if err != nil {
		return agent.MessagesResponse{}, err
	}
	tools, err := AgentToolsToGenAISchema(req.Tools)
	if err != nil {
		return agent.MessagesResponse{}, err
	}

	config := &genai.GenerateContentConfig{
		CandidateCount: int32(1),
		SystemInstruction: &genai.Content{
			Parts: []*genai.Part{
				{
					Text: req.System,
				},
			},
		},
		Tools: tools,
	}

	if req.MaxTokens > 0 {
		config.MaxOutputTokens = int32(req.MaxTokens)
	}

	resp, err := c.client.Models.GenerateContent(ctx, c.model, msgs, config)
	if err != nil {
		return agent.MessagesResponse{}, err
	}

	return c.convertMessagesResponse(resp)
}

func (c *Client) convertMessagesResponse(resp *genai.GenerateContentResponse) (agent.MessagesResponse, error) {
	var out agent.MessagesResponse
	if len(resp.Candidates) == 0 {
		return out, errors.New("no candidates returned")
	}
	gc := resp.Candidates[0]

	// Check if the response was truncated due to MaxOutputTokens limit
	if gc.FinishReason == genai.FinishReasonMaxTokens {
		return out, fmt.Errorf("%w: %v", agent.ErrTooLarge, gc.FinishReason)
	}
	for _, p := range gc.Content.Parts {
		switch {
		case p.FunctionCall != nil:
			b, err := json.Marshal(p.FunctionCall.Args)
			if err != nil {
				return out, err
			}
			out.Content = append(out.Content, agent.MessageContent{
				Type:    agent.ContentTypeToolUse,
				ToolUse: &agent.ToolUse{ID: p.FunctionCall.ID, Name: p.FunctionCall.Name, Input: b},
			})
		case p.Text != "":
			out.Content = append(out.Content, agent.MessageContent{
				Type: agent.ContentTypeText,
				Text: p.Text,
			})
		}
	}
	if u := resp.UsageMetadata; u != nil {
		if u.CandidatesTokenCount != 0 {
			out.Usage.OutputTokens = int(u.CandidatesTokenCount)
		}
		if u.CachedContentTokenCount != 0 {
			out.Usage.CacheReadInputTokens = int(u.CachedContentTokenCount)
		}
		if u.PromptTokenCount != 0 {
			out.Usage.InputTokens = int(u.PromptTokenCount) - out.Usage.CacheReadInputTokens
		}
		// TODO: how do we actually create the cache?
	}

	// Set stop reason using raw finish reason
	out.StopReason = string(gc.FinishReason)

	return out, nil
}
