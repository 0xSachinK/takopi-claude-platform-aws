# takopi-claude-platform-aws

Takopi engine plugin that routes prompts through the Anthropic Python SDK instead
of a shell CLI. It uses Claude Platform on AWS when a workspace identifier is
configured, falls back to a direct Anthropic API key when present, and can use
AWS Bedrock as the primary or fallback provider.

Takopi 0.22 validates engine plugin IDs with underscores. The package name is
`takopi-claude-platform-aws`, but the engine ID is `claude_platform_aws`.

## Install

```sh
pip install takopi-claude-platform-aws
```

For a Git checkout:

```sh
pip install .
```

Validate discovery:

```sh
takopi plugins --load
```

## Configuration

Minimal `~/.takopi/takopi.toml`:

```toml
default_engine = "claude_platform_aws"

[claude_platform_aws]
workspace_id = "your-workspace-id"
region = "us-east-1"
primary_model = "claude-opus-4-6"
fallback_model = "claude-opus-4-6"
max_tokens = 8000
max_iterations = 25
retry_count = 2
```

The plugin also reads `[engines.claude_platform_aws]` and
`[engines."claude-platform-aws"]` for operators who prefer grouped engine
settings, but Takopi still needs `default_engine = "claude_platform_aws"`.

## Worked Example

```toml
default_engine = "claude_platform_aws"

[claude_platform_aws]
workspace_id = "your-workspace-id"
region = "us-east-1"
primary_model = "claude-opus-4-6"
fallback_model = "claude-opus-4-6"
workspace_root = "/srv/takopi/workspace"
skills_dir = "/srv/takopi/skills"
kb_dir = "/srv/takopi/kb"
mcp_config = "/srv/takopi/mcp.json"
enabled_tools = ["Bash", "Read", "Write", "Edit", "Grep", "Glob", "LS"]
stream_text = true
```

Example MCP config:

```json
{
  "mcpServers": {
    "docs": {
      "command": "node",
      "args": ["/srv/takopi/mcp/docs-server.js"],
      "env": {
        "TOKEN": "${DOCS_TOKEN}"
      }
    }
  }
}
```

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `TAKOPI_CLAUDE_PLATFORM_AWS_WORKSPACE_ID` | Claude Platform on AWS workspace identifier. |
| `ANTHROPIC_AWS_WORKSPACE_ID` | Alternate workspace identifier name. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_REGION` | AWS region override. |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | Region fallback used by AWS SDK clients. |
| `AWS_ACCESS_KEY_ID` | Optional explicit AWS access key. |
| `AWS_SECRET_ACCESS_KEY` | Optional explicit AWS secret key. |
| `AWS_SESSION_TOKEN` | Optional temporary AWS session token. |
| `ANTHROPIC_API_KEY` | Direct Anthropic API fallback when no workspace identifier is set. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_MODEL` | Primary model alias. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_PRIMARY_MODEL` | Primary model alias. |
| `ANTHROPIC_MODEL` | Primary model fallback. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_FALLBACK_MODEL` | Bedrock fallback model alias. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_MAX_TOKENS` | Maximum output tokens per API call. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_MAX_ITERATIONS` | Maximum tool-use round trips. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_RETRY_COUNT` | Primary provider retries before Bedrock fallback. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_RETRY_BASE_DELAY_S` | Exponential backoff base delay. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_WORKSPACE_ROOT` | Root for local tools. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_SESSION_STORE` | JSON session history path. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_SKILLS_DIR` | Directory scanned for `SKILL.md` files. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_KB_DIR` | Directory scanned for Markdown knowledge files. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_MCP_CONFIG` | MCP server JSON config path. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_ENABLED_TOOLS` | Comma-separated local tools. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_BASH_TIMEOUT_S` | Default Bash tool timeout. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_ALLOW_OUTSIDE_WORKSPACE` | Allow file tools outside the workspace root. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_TOOL_RESULT_LIMIT` | Maximum serialized tool result size. |
| `TAKOPI_CLAUDE_PLATFORM_AWS_STREAM_TEXT` | Stream partial text as Takopi progress events. |

## Shipped Behavior

- Auth resolver: Claude Platform on AWS, direct Anthropic API, then Bedrock.
- Streaming through Takopi started/action/completed events.
- Local tools: Bash, Read, Write, Edit, Grep, Glob, LS.
- MCP bridge exposing tools as `mcp__server__tool`.
- Skills and knowledge-base prompt indexes with Anthropic prompt caching.
- JSON session history keyed by Takopi resume token.
- Bedrock fallback after transient 5xx, overload, connection, or credential
  validation failures.
- Final answer footer showing the actual model and provider.

## Credits

Built for the Takopi plugin system created by banteg/takopi.
