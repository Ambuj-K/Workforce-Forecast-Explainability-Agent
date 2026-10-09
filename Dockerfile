# Workforce Forecast Explainability Agent: one image for any container platform.
#
#   docker build -t wfx .
#   docker run -p 8000:8000 \
#     -e WFX_USERS_FILE=/run/config/users.yaml -v $PWD/users.yaml:/run/config/users.yaml:ro \
#     -e GOOGLE_API_KEY_FILE=/run/secrets/google_api_key -v $PWD/key.txt:/run/secrets/google_api_key:ro \
#     --read-only --tmpfs /tmp wfx
#
# Sign-in is on by default (WFX_AUTH=token). The LLM key and the users file are mounted at run
# time, never baked in. The agent database is built from synthetic data in a builder stage, so
# the image is self-contained and reproducible (fixed seed).

ARG PYTHON=3.11-slim

# ---- dependencies: locked, no dev tools, project installed as a regular package
FROM python:${PYTHON} AS deps
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev --no-editable

# ---- data: generate the synthetic history, train, back-test, build the locked agent database
FROM deps AS data
COPY scripts ./scripts
RUN .venv/bin/python scripts/build_extract.py --out /build/extract --db /build/agent.duckdb

# ---- runtime: no compiler, no uv, no source tree, non-root, read-only friendly
FROM python:${PYTHON} AS runtime
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin wfx
WORKDIR /app
COPY --from=deps /app/.venv ./.venv
COPY scripts/serve.py ./scripts/serve.py
COPY --from=data /build/agent.duckdb ./data/agent.duckdb
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    WFX_HOST=0.0.0.0 PORT=8000 \
    WFX_AUTH=token \
    WFX_DB=/app/data/agent.duckdb \
    WFX_LOG=/tmp/wfx/interactions.jsonl
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=2)"
CMD ["python", "scripts/serve.py"]
