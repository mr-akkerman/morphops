FROM python:3.12-slim

# System deps: nodejs/npm for npx-based MCP servers, curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs npm curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv from official image (faster than pip install uv)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install Python deps first (cached layer — only rebuilds on pyproject.toml change)
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# Copy source (split to maximise cache hits)
COPY config.py ./
COPY agent/ ./agent/
COPY tools/ ./tools/
COPY bot/ ./bot/

# Runtime dirs — overridden by volume mounts in production
RUN mkdir -p agent_tools agent_mcp agent_context agent_tmp

# Drop root privileges
RUN useradd -r -u 1000 -m botuser && chown -R botuser:botuser /app
USER botuser

# Liveness check: ensure the bot process is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD pgrep -f "bot.main" > /dev/null || exit 1

CMD ["uv", "run", "python", "-m", "bot.main"]
