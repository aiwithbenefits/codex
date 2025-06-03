use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::LazyLock;

use crate::client_common::Prompt;
use crate::config_types::McpServerConfig;

pub type McpToolHeaders = BTreeMap<String, String>;

#[derive(Debug, Clone, Deserialize)]
pub enum McpRequireApproval {
    Never,
    // 'Always' can be explicitly set or be the default if require_approval is omitted.
    // If this variant is present in Option<McpRequireApproval>, it serializes to "always".
    Always,
    SelectiveNever { tool_names: Vec<String> },
}

// Custom serialization for McpRequireApproval
impl Serialize for McpRequireApproval {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            McpRequireApproval::Never => serializer.serialize_str("never"),
            McpRequireApproval::Always => serializer.serialize_str("always"),
            McpRequireApproval::SelectiveNever { tool_names } => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(1))?;
                // Construct the inner {"tool_names": ["name1", "name2"]} object
                let mut inner_object = BTreeMap::new();
                inner_object.insert("tool_names".to_string(), tool_names.clone());
                map.serialize_entry("never", &inner_object)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for McpRequireApproval {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct McpRequireApprovalVisitor;

        impl<'de> serde::de::Visitor<'de> for McpRequireApprovalVisitor {
            type Value = McpRequireApproval;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a string (never, always) or an object like { \"never\": {\"tool_names\": []} } for selective approval")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "never" => Ok(McpRequireApproval::Never),
                    "always" => Ok(McpRequireApproval::Always),
                    _ => Err(E::custom(format!("Unknown string variant for McpRequireApproval: {}", value))),
                }
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                #[derive(Deserialize)]
                struct SelectiveNeverHelper {
                    tool_names: Vec<String>,
                }

                // Expect a single key "never"
                if let Some(key) = map.next_key::<String>()? {
                    if key == "never" {
                        let val: SelectiveNeverHelper = map.next_value()?;
                        // Check if there are more keys, which would be an error
                        if map.next_key::<String>()?.is_some() {
                            return Err(serde::de::Error::custom("Too many keys in McpRequireApproval map, expected only 'never'"));
                        }
                        return Ok(McpRequireApproval::SelectiveNever { tool_names: val.tool_names });
                    } else {
                        return Err(serde::de::Error::custom(format!("Expected key 'never' in McpRequireApproval map, got '{}'", key)));
                    }
                }
                Err(serde::de::Error::custom("Expected map with a 'never' key for selective approval for McpRequireApproval"))
            }
        }
        deserializer.deserialize_any(McpRequireApprovalVisitor)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct McpToolSpec {
    pub server_label: String,
    pub server_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_approval: Option<McpRequireApproval>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<McpToolHeaders>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ResponsesApiTool {
    name: &'static str,
    description: &'static str,
    strict: bool,
    parameters: JsonSchema,
}

/// When serialized as JSON, this produces a valid "Tool" in the OpenAI
/// Responses API.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub(crate) enum OpenAiTool {
    #[serde(rename = "function")]
    Function(ResponsesApiTool),
    #[serde(rename = "local_shell")]
    LocalShell {},
    #[serde(rename = "mcp")]
    Mcp(McpToolSpec),
}

/// Generic JSON‑Schema subset needed for our tool definitions
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub(crate) enum JsonSchema {
    String,
    Number,
    Array {
        items: Box<JsonSchema>,
    },
    Object {
        properties: BTreeMap<String, JsonSchema>,
        required: &'static [&'static str],
        #[serde(rename = "additionalProperties")]
        additional_properties: bool,
    },
}

/// Tool usage specification
static DEFAULT_TOOLS: LazyLock<Vec<OpenAiTool>> = LazyLock::new(|| {
    let mut properties = BTreeMap::new();
    properties.insert(
        "command".to_string(),
        JsonSchema::Array {
            items: Box::new(JsonSchema::String),
        },
    );
    properties.insert("workdir".to_string(), JsonSchema::String);
    properties.insert("timeout".to_string(), JsonSchema::Number);

    vec![OpenAiTool::Function(ResponsesApiTool {
        name: "shell",
        description: "Runs a shell command, and returns its output.",
        strict: false,
        parameters: JsonSchema::Object {
            properties,
            required: &["command"],
            additional_properties: false,
        },
    })]
});

static DEFAULT_CODEX_MODEL_TOOLS: LazyLock<Vec<OpenAiTool>> =
    LazyLock::new(|| vec![OpenAiTool::LocalShell {}]);

/// Returns JSON values that are compatible with Function Calling in the
/// Responses API:
/// https://platform.openai.com/docs/guides/function-calling?api-mode=responses
pub(crate) fn create_tools_json_for_responses_api(
    prompt: &Prompt,
    model: &str,
) -> crate::error::Result<Vec<serde_json::Value>> {
    // Assemble tool list: built-in tools + any extra tools from the prompt.
    let default_tools = if model.starts_with("codex") {
        &DEFAULT_CODEX_MODEL_TOOLS
    } else {
        &DEFAULT_TOOLS
    };

    let mut tools_json = Vec::with_capacity(default_tools.len() + prompt.mcp_configs.as_ref().map_or(0, |cfgs| cfgs.len()));

    for t in default_tools.iter() {
        tools_json.push(serde_json::to_value(t)?);
    }

    if let Some(mcp_configs_map_arc) = &prompt.mcp_configs {
        for (server_label, mcp_conf) in mcp_configs_map_arc.iter() {
            if let Some(server_url) = &mcp_conf.server_url {
                let spec = McpToolSpec {
                    server_label: server_label.clone(),
                    server_url: server_url.clone(),
                    require_approval: mcp_conf.require_approval.clone(),
                    allowed_tools: mcp_conf.allowed_tools.clone(),
                    headers: mcp_conf.headers.clone(),
                };
                let mcp_tool_value = serde_json::to_value(OpenAiTool::Mcp(spec))?;
                tools_json.push(mcp_tool_value);
            }
            // Optional: Else, if no server_url, these are local/stdio MCP servers.
            // Their tools are currently expected to be in prompt.extra_tools
            // and handled by the mcp_tool_to_openai_tool pathway if that block is restored/modified.
        }
    }

    // The existing logic for prompt.extra_tools is now commented out as per instructions.
    // tools_json.extend(
    //     prompt
    //         .extra_tools
    //         .clone()
    //         .into_iter()
    //         .map(|(name, tool)| mcp_tool_to_openai_tool(name, tool)),
    // );

    Ok(tools_json)
}

/// Returns JSON values that are compatible with Function Calling in the
/// Chat Completions API:
/// https://platform.openai.com/docs/guides/function-calling?api-mode=chat
pub(crate) fn create_tools_json_for_chat_completions_api(
    prompt: &Prompt,
    model: &str,
) -> crate::error::Result<Vec<serde_json::Value>> {
    // We start with the JSON for the Responses API and than rewrite it to match
    // the chat completions tool call format.
    let responses_api_tools_json = create_tools_json_for_responses_api(prompt, model)?;
    let tools_json = responses_api_tools_json
        .into_iter()
        .filter_map(|mut tool| {
            if tool.get("type") != Some(&serde_json::Value::String("function".to_string())) {
                return None;
            }

            if let Some(map) = tool.as_object_mut() {
                // Remove "type" field as it is not needed in chat completions.
                map.remove("type");
                Some(json!({
                    "type": "function",
                    "function": map,
                }))
            } else {
                None
            }
        })
        .collect::<Vec<serde_json::Value>>();
    Ok(tools_json)
}

fn mcp_tool_to_openai_tool(
    fully_qualified_name: String,
    tool: mcp_types::Tool,
) -> serde_json::Value {
    let mcp_types::Tool {
        description,
        mut input_schema,
        ..
    } = tool;

    // OpenAI models mandate the "properties" field in the schema. The Agents
    // SDK fixed this by inserting an empty object for "properties" if it is not
    // already present https://github.com/openai/openai-agents-python/issues/449
    // so here we do the same.
    if input_schema.properties.is_none() {
        input_schema.properties = Some(serde_json::Value::Object(serde_json::Map::new()));
    }

    // TODO(mbolin): Change the contract of this function to return
    // ResponsesApiTool.
    json!({
        "name": fully_qualified_name,
        "description": description,
        "parameters": input_schema,
        "type": "function",
    })
}
