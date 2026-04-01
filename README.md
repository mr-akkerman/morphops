# AI Admin Bot

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker-compose.yml)
[![Powered by Claude](https://img.shields.io/badge/powered%20by-Claude-orange.svg)](https://anthropic.com)

**A Telegram bot that becomes the admin of anything you connect to it.**

Connect a PostgreSQL database, paste some API docs, or point it at an MCP server — the bot writes its own tools and starts managing it. No code changes. No restarts. Just describe what you need in chat.

## What makes it different

| | |
|---|---|
| **Self-extends** | Writes Python tools on the fly and hot-reloads them immediately |
| **DB admin** | Connects to PostgreSQL, introspects schema, generates query tools |
| **API admin** | Paste any API docs — it writes HTTP wrappers with auth pre-filled |
| **MCP-ready** | Installs and connects any MCP server (`uvx` / `npx`) on demand |
| **Transparent** | Each reasoning step appears as a separate Telegram message |

## Quick Start

**Prerequisites:** Docker + Docker Compose, Telegram bot token (from [@BotFather](https://t.me/BotFather)), Anthropic API key.

```bash
git clone https://github.com/mr-akkerman/ai-admin
cd ai-admin
cp .env.example .env        # add your ANTHROPIC_API_KEY and TELEGRAM_BOT_TOKEN
docker compose up -d
```

Send `/start` to your bot — 18 built-in tools are registered and ready.

## Example session

```
You:  Connect to my DB: postgresql://user:pass@host:5432/prod
Bot:  ✅ Connected. Schema loaded — 12 tables found.
      ✅ Generated tools: query_users, query_orders, query_products

You:  Show me all users who signed up this week
Bot:  🔧 execute_raw_sql(...)
      Found 47 users. Here's the summary...
```

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | — | Anthropic API key |
| `TELEGRAM_BOT_TOKEN` | ✅ | — | Token from @BotFather |
| `ALLOWED_USER_IDS` | — | *(everyone)* | Comma-separated Telegram user IDs |
| `LLM_MODEL` | — | `claude-sonnet-4-6` | Claude model ID |
| `MAX_AGENT_ITERATIONS` | — | `15` | Max reasoning steps per message |
| `MAX_TOOL_EXECUTION_TIME` | — | `30` | Tool execution timeout (seconds) |
| `DB_READONLY` | — | `false` | Restrict DB to SELECT-only |

## Architecture

```
Telegram message
    └─▶ ReAct loop (agent/loop.py)
            ├─▶ Built-in tools (tools/)
            │       ├── meta_tools.py   — write_tool, save_env_var, MCP management
            │       ├── db_tools.py     — PostgreSQL connection + SQL execution
            │       └── api_tools.py    — HTTP GET / POST / PATCH / DELETE
            └─▶ Agent-written tools (agent_tools/*.py)  ← hot-loaded at runtime
```

Runtime directories — Docker volumes, survive restarts and rebuilds:

| Directory | Contents |
|---|---|
| `agent_tools/` | Python tools written by the agent |
| `agent_context/` | Markdown notes (schemas, API docs, custom instructions) |
| `agent_mcp/` | MCP server configs (`servers.json`) |
| `agent_tmp/` | Temporary files (uploads, SQL dumps) |

## Development

```bash
pip install uv
uv sync
cp .env.example .env
uv run python -m bot.main
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
