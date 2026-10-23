# websearch MCP server — container image
#
# Build:  docker build -t websearch-mcp .
# Run:    docker run --rm -i websearch-mcp
#
# The server speaks MCP over stdio, so it is meant to be attached to an MCP
# client (Claude Code, Cursor, Codex, …) rather than exposed on a port.
#
# Optional API keys are passed as environment variables at RUN time, never
# baked into the image:
#   docker run --rm -i -e GROQ_API_KEYS=... websearch-mcp
#
# The heavy optional tiers are NOT installed by default, to keep the image
# small. Add them with --build-arg EXTRAS="[all]".
FROM python:3.12-slim

ARG EXTRAS=""

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the published package into a normal site-packages location.
RUN pip install --no-cache-dir "skdev-websearch-mcp${EXTRAS}"

# Run as a non-root user.
RUN useradd --create-home --uid 10001 mcp
USER mcp

# stdio transport: no ports, no writes outside the home cache dir.
ENTRYPOINT ["websearch-mcp"]
