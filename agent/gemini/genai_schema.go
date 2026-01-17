// Copyright (c) 2025 AUTHORS All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gemini

import (
	"encoding/json"
	"reflect"

	"github.com/invopop/jsonschema"
	"google.golang.org/genai"
	"pkg.maisem.dev/agent"
	"tailscale.com/types/ptr"
)

// GenAISchema converts a Go type to a genai.Schema representation.
// This is the equivalent of JSONSchema but for genai.Schema.
func GenAISchema[T any]() *genai.Schema {
	return TypeToGenAISchema(reflect.TypeFor[T]())
}

// TypeToGenAISchema converts a reflect.Type to a genai.Schema.
func TypeToGenAISchema(t reflect.Type) *genai.Schema {
	// First convert to JSON schema using the existing method
	rt := jsonschema.Reflector{
		DoNotReference: true,
		BaseSchemaID:   "agents",
	}
	jsonSchema := rt.ReflectFromType(t)

	// Convert JSON schema to genai.Schema
	return ConvertJSONSchemaToGenAISchema(jsonSchema)
}

func fromJSONNumber(n json.Number) *float64 {
	f, err := n.Float64()
	if err != nil {
		return nil
	}
	return &f
}

func uint64ToInt64(u *uint64) *int64 {
	if u == nil {
		return nil
	}
	return ptr.To(int64(*u))
}

// ConvertJSONSchemaToGenAISchema converts a jsonschema.Schema to a genai.Schema.
func ConvertJSONSchemaToGenAISchema(js *jsonschema.Schema) *genai.Schema {
	if js == nil {
		return nil
	}

	schema := &genai.Schema{
		Title:         js.Title,
		Description:   js.Description,
		Type:          mapJSONSchemaTypeToGenAIType(js.Type),
		Format:        js.Format,
		Required:      js.Required,
		Items:         ConvertJSONSchemaToGenAISchema(js.Items),
		Default:       js.Default,
		Minimum:       fromJSONNumber(js.Minimum),
		Maximum:       fromJSONNumber(js.Maximum),
		Pattern:       js.Pattern,
		MinLength:     uint64ToInt64(js.MinLength),
		MaxLength:     uint64ToInt64(js.MaxLength),
		MaxItems:      uint64ToInt64(js.MaxItems),
		MinItems:      uint64ToInt64(js.MinItems),
		MinProperties: uint64ToInt64(js.MinProperties),
		MaxProperties: uint64ToInt64(js.MaxProperties),

		// filled below
		Properties: nil,
		Enum:       nil,
		AnyOf:      nil,

		// Nil...
		Example:          "",
		Nullable:         nil,
		PropertyOrdering: nil,
	}

	if len(js.AnyOf) > 0 {
		schema.AnyOf = make([]*genai.Schema, 0, len(js.AnyOf))
		for _, anyOf := range js.AnyOf {
			schema.AnyOf = append(schema.AnyOf, ConvertJSONSchemaToGenAISchema(anyOf))
		}
	}

	// Handle enum values for string type
	if len(js.Enum) > 0 && schema.Type == genai.TypeString {
		schema.Enum = make([]string, 0, len(js.Enum))
		for _, enumVal := range js.Enum {
			if strVal, ok := enumVal.(string); ok {
				schema.Enum = append(schema.Enum, strVal)
			}
		}
	}

	// Handle object properties
	if js.Properties != nil {
		schema.Properties = make(map[string]*genai.Schema)
		iter := js.Properties.Oldest()
		for iter != nil {
			schema.Properties[iter.Key] = ConvertJSONSchemaToGenAISchema(iter.Value)
			iter = iter.Next()
		}
	}

	return schema
}

// mapJSONSchemaTypeToGenAIType maps jsonschema type to genai.Type.
func mapJSONSchemaTypeToGenAIType(jsType string) genai.Type {
	switch jsType {
	case "string":
		return genai.TypeString
	case "number":
		return genai.TypeNumber
	case "integer":
		return genai.TypeInteger
	case "boolean":
		return genai.TypeBoolean
	case "array":
		return genai.TypeArray
	case "object":
		return genai.TypeObject
	default:
		return genai.TypeString // Default to string for unknown types
	}
}

// ToolToGenAISchema converts a Tool to a genai.Schema.
func ToolToGenAISchema(tool agent.ToolDefinition) (*genai.Schema, error) {
	var js jsonschema.Schema
	if err := json.Unmarshal([]byte(tool.InputSchema), &js); err != nil {
		return nil, err
	}
	schema := ConvertJSONSchemaToGenAISchema(&js)
	return schema, nil
}

// AgentToolsToGenAISchema converts a map of tools to a list of function definitions
// using genai.Schema for the function parameters.
func AgentToolsToGenAISchema(tools []agent.ToolDefinition) ([]*genai.Tool, error) {
	out := make([]*genai.Tool, 0, len(tools))

	for _, tool := range tools {
		paramSchema, err := ToolToGenAISchema(tool)
		if err != nil {
			return nil, err
		}

		// Create function declaration
		funcDecl := &genai.FunctionDeclaration{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  paramSchema,
		}

		out = append(out, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{funcDecl},
		})
	}

	return out, nil
}
