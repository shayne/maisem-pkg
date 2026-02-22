// Copyright (c) 2025 AUTHORS All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package models contains constants for the models used in the application.
package models

import "fmt"

const (
	Claude3Opus20240229 = "claude-3-opus-20240229"
	Claude3OpusLatest   = Claude3Opus20240229

	Claude3Dot5Haiku20241022 = "claude-3-5-haiku-20241022"
	Claude3Dot5HaikuLatest   = Claude3Dot5Haiku20241022

	Claude3Dot7Sonnet20250219 = "claude-3-7-sonnet-20250219"
	Claude3Dot7SonnetLatest   = Claude3Dot7Sonnet20250219

	Claude3Dot5Sonnet20241022 = "claude-3-5-sonnet-20241022"
	Claude3Dot5SonnetLatest   = Claude3Dot5Sonnet20241022

	Claude4Opus20250514  = "claude-opus-4-20250514"
	Claude4OpusLatest    = Claude4Opus20250514
	Claude41Opus20250805 = "claude-opus-4-1-20250805"
	Claude41OpusLatest   = Claude41Opus20250805

	Claude4Sonnet20250514 = "claude-sonnet-4-20250514"
	Claude4SonnetLatest   = Claude4Sonnet20250514

	GPT5        = "gpt-5"
	GPT52       = "gpt-5.2"
	GPT52Latest = GPT52

	DefaultModel      = Claude3Dot5SonnetLatest
	DefaultSmallModel = Claude3Dot5HaikuLatest

	VertexClaude3Dot7Sonnet20250219 = "claude-3-7-sonnet@20250219"
	VertexClaude3Dot7SonnetLatest   = VertexClaude3Dot7Sonnet20250219
	VertexClaude3Dot5Sonnet20241022 = "claude-3-5-sonnet-v2@20241022"
	VertexClaude3Dot5SonnetLatest   = VertexClaude3Dot5Sonnet20241022

	O320250416 = "o3-2025-04-16"
	O3Latest   = O320250416

	Qwen3Coder480b   = "qwen/qwen3-coder"
	Qwen3CoderLatest = Qwen3Coder480b

	Gemini2Dot5Pro   = "gemini-2.5-pro"
	Gemini2Dot5Flash = "gemini-2.5-flash"
)

func AsVertex(model string) (string, error) {
	switch model {
	case Claude3Dot7SonnetLatest:
		return VertexClaude3Dot7SonnetLatest, nil
	case Claude3Dot5SonnetLatest:
		return VertexClaude3Dot5SonnetLatest, nil
	}
	return "", fmt.Errorf("unknown model: %s", model)
}
