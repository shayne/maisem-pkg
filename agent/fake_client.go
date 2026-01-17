// Copyright (c) 2025 AUTHORS All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// FakeLLMClient provides a configurable fake LLM client for testing.
// It can simulate various LLM behaviors including tool calls, errors, and delays.
//
// FakeLLMClient is more feature-rich than mockLLMClient and provides:
//   - Custom request handlers via OnRequest
//   - Response delays to simulate network latency
//   - Request history tracking
//   - Builder pattern for easy configuration
//   - Default responses when configured responses are exhausted
//   - Token counting support
//
// Usage with builder:
//
//	client := NewFakeLLMClient().
//	    WithTextResponse("Hello").
//	    WithToolCallResponse("my_tool", map[string]any{"arg": "value"}).
//	    WithError(errors.New("rate limited")).
//	    WithDelay(100 * time.Millisecond).
//	    Build()
//
// Usage with custom handler:
//
//	client := &FakeLLMClient{
//	    OnRequest: func(ctx context.Context, req MessagesRequest) (*MessagesResponse, error) {
//	        // Custom logic based on request
//	        return &MessagesResponse{...}, nil
//	    },
//	}
type FakeLLMClient struct {
	mu sync.Mutex

	// Responses to return in sequence
	responses []MessagesResponse

	// Errors to return in sequence (parallel to responses)
	errors []error

	// Current call count
	callCount int

	// RequestHistory stores all requests made to this client
	RequestHistory []MessagesRequest

	// ResponseDelay adds artificial delay to simulate network latency
	ResponseDelay time.Duration

	// DefaultResponse is used when responses slice is exhausted
	DefaultResponse *MessagesResponse

	// DefaultError is used when errors slice is exhausted (after DefaultResponse)
	DefaultError error

	// TokenCounter function for custom token counting logic
	TokenCounter func(req MessagesRequest) int

	// OnRequest is called for each request, allowing custom behavior
	OnRequest func(ctx context.Context, req MessagesRequest) (*MessagesResponse, error)
}

// CreateMessages implements the LLMClient interface
func (f *FakeLLMClient) CreateMessages(ctx context.Context, req MessagesRequest, onUpdate func(MessagesResponse)) (MessagesResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	// Store request in history
	f.RequestHistory = append(f.RequestHistory, req)

	// Simulate delay if configured
	if f.ResponseDelay > 0 {
		select {
		case <-time.After(f.ResponseDelay):
		case <-ctx.Done():
			return MessagesResponse{}, ctx.Err()
		}
	}

	// Call custom handler if provided
	if f.OnRequest != nil {
		resp, err := f.OnRequest(ctx, req)
		f.callCount++
		if resp != nil {
			return *resp, err
		}
		// If handler returns nil response with error, return the error
		if err != nil {
			return MessagesResponse{}, err
		}
		// If handler returns nil response without error, continue to default behavior
	}

	// Return configured response/error
	if f.callCount < len(f.responses) {
		resp := f.responses[f.callCount]
		var err error
		if f.callCount < len(f.errors) {
			err = f.errors[f.callCount]
		}
		f.callCount++
		return resp, err
	}

	// Use default response if available
	if f.DefaultResponse != nil {
		f.callCount++
		return *f.DefaultResponse, f.DefaultError
	}

	// Generate a simple text response as fallback
	f.callCount++
	return MessagesResponse{
		ID:         fmt.Sprintf("fake_%d", f.callCount),
		StopReason: "end_turn",
		Content: Content{
			{
				Type: ContentTypeText,
				Text: "Default response",
			},
		},
		Usage: Usage{
			InputTokens:  100,
			OutputTokens: 50,
		},
	}, nil
}

// CountTokens implements the TokenCounter interface
func (f *FakeLLMClient) CountTokens(ctx context.Context, req MessagesRequest) (int, error) {
	if f.TokenCounter != nil {
		return f.TokenCounter(req), nil
	}
	// Simple approximation: count characters / 4
	total := len(req.System) / 4
	for _, msg := range req.Messages {
		for _, content := range msg.Content {
			total += len(content.GetText()) / 4
		}
	}
	return total, nil
}

// Reset clears the call count and request history
func (f *FakeLLMClient) Reset() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.callCount = 0
	f.RequestHistory = nil
}

// GetCallCount returns the number of times CreateMessages was called
func (f *FakeLLMClient) GetCallCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.callCount
}

// GetLastRequest returns the most recent request, or nil if no requests
func (f *FakeLLMClient) GetLastRequest() *MessagesRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.RequestHistory) == 0 {
		return nil
	}
	return &f.RequestHistory[len(f.RequestHistory)-1]
}

// FakeLLMClientBuilder provides a fluent interface for building FakeLLMClient instances
type FakeLLMClientBuilder struct {
	client *FakeLLMClient
}

// NewFakeLLMClient creates a new builder for FakeLLMClient
func NewFakeLLMClient() *FakeLLMClientBuilder {
	return &FakeLLMClientBuilder{
		client: &FakeLLMClient{},
	}
}

// WithResponse adds a response to return
func (b *FakeLLMClientBuilder) WithResponse(resp MessagesResponse) *FakeLLMClientBuilder {
	b.client.responses = append(b.client.responses, resp)
	return b
}

// WithTextResponse adds a simple text response
func (b *FakeLLMClientBuilder) WithTextResponse(text string) *FakeLLMClientBuilder {
	return b.WithResponse(MessagesResponse{
		Content: Content{
			{Type: ContentTypeText, Text: text},
		},
		StopReason: "end_turn",
	})
}

// WithToolCallResponse adds a response with a tool call
func (b *FakeLLMClientBuilder) WithToolCallResponse(toolName string, args map[string]any) *FakeLLMClientBuilder {
	argsJSON, _ := json.Marshal(args)
	return b.WithResponse(MessagesResponse{
		Content: Content{
			{
				Type: ContentTypeToolUse,
				ToolUse: &ToolUse{
					ID:    fmt.Sprintf("call_%d", len(b.client.responses)+1),
					Name:  toolName,
					Input: argsJSON,
				},
			},
		},
		StopReason: "tool_use",
	})
}

// WithError adds an error to return at the corresponding call
func (b *FakeLLMClientBuilder) WithError(err error) *FakeLLMClientBuilder {
	// Pad errors slice if needed
	for len(b.client.errors) < len(b.client.responses)-1 {
		b.client.errors = append(b.client.errors, nil)
	}
	b.client.errors = append(b.client.errors, err)
	return b
}

// WithDelay adds response delay
func (b *FakeLLMClientBuilder) WithDelay(d time.Duration) *FakeLLMClientBuilder {
	b.client.ResponseDelay = d
	return b
}

// WithDefaultResponse sets the default response when configured responses are exhausted
func (b *FakeLLMClientBuilder) WithDefaultResponse(resp MessagesResponse) *FakeLLMClientBuilder {
	b.client.DefaultResponse = &resp
	return b
}

// WithDefaultError sets the default error when configured responses are exhausted
func (b *FakeLLMClientBuilder) WithDefaultError(err error) *FakeLLMClientBuilder {
	b.client.DefaultError = err
	return b
}

// WithOnRequest sets a custom request handler
func (b *FakeLLMClientBuilder) WithOnRequest(fn func(context.Context, MessagesRequest) (*MessagesResponse, error)) *FakeLLMClientBuilder {
	b.client.OnRequest = fn
	return b
}

// Build returns the configured FakeLLMClient
func (b *FakeLLMClientBuilder) Build() *FakeLLMClient {
	return b.client
}
