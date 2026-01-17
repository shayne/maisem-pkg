// Copyright (c) 2025 AUTHORS All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package perplexity

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/carlmjohnson/requests"
)

const defaultAPIURL = "https://api.perplexity.ai"

// Client represents a Perplexity API client.
type Client struct {
	apiKey string
	apiURL string
}

// New creates a new Perplexity client with the given API key.
func New(apiKey string) *Client {
	return &Client{
		apiKey: apiKey,
		apiURL: defaultAPIURL,
	}
}

// WithURL returns a new client with a custom API URL.
func (c *Client) WithURL(url string) *Client {
	c2 := *c
	c2.apiURL = strings.TrimSuffix(url, "/")
	return &c2
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type CompletionRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

type CompletionResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Created int64  `json:"created"`
	Choices []struct {
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// Query sends a query to Perplexity and returns the first response.
func (c *Client) Query(ctx context.Context, query string) (string, error) {
	req := CompletionRequest{
		Model: "sonar",
		Messages: []Message{
			{Role: "user", Content: query},
		},
	}

	var resp CompletionResponse
	err := requests.
		URL(c.apiURL+"/chat/completions").
		Method(http.MethodPost).
		Header("accept", "application/json").
		Header("content-type", "application/json").
		Bearer(c.apiKey).
		BodyJSON(&req).
		ToJSON(&resp).
		Fetch(ctx)
	if err != nil {
		return "", fmt.Errorf("error querying perplexity: %w", err)
	}

	return resp.Choices[0].Message.Content, nil
}
