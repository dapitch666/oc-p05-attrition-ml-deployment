FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first, so this layer is cached until uv.lock changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# Non-editable install: the runtime stage needs no source tree, and the package
# metadata (read for the API version) is installed in the virtual environment.
COPY src/ src/
RUN uv sync --locked --no-dev --no-editable

# Bytecode is not compiled and the test suites of third-party packages are dropped: together
# they weigh about 290 MB in the image and are not needed to serve predictions.
RUN find .venv -type d -name tests -prune -exec rm -rf {} +

FROM python:3.12-slim-bookworm
RUN useradd --uid 1000 --create-home app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY models/ models/
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "oc_p05_attrition_ml_deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
