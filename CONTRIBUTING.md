# Contributing

## Setup

```bash
git clone https://github.com/mr-akkerman/ai-admin
cd ai-admin
pip install uv
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY and TELEGRAM_BOT_TOKEN
uv run python -m bot.main
```

## Project structure

```
agent/      — ReAct loop, tool registry, MCP client
bot/        — Telegram handlers and message transport
tools/      — Built-in tools (meta, DB, HTTP)
config.py   — Environment config via python-dotenv
docs/       — Architecture, security audit, implementation notes
```

## Submitting changes

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

## Guidelines

- Keep PRs focused — one change per PR
- Match the existing code style
- Update `README.md` and `.env.example` if you add new env vars
- Built-in tools live in `tools/` — each file is one domain (`meta`, `db`, `http`)

## Reporting issues

Open a [GitHub issue](https://github.com/mr-akkerman/ai-admin/issues) and include:

- What you expected to happen
- What actually happened
- Steps to reproduce
