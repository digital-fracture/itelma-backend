FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV UV_NO_DEV=true
ENV UV_LOCKED=true
ENV UV_COMPILE_BYTECODE=true
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/cache/uv

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=$UV_CACHE_DIR \
    uv sync --no-install-project

COPY app/ ./app
RUN --mount=type=cache,target=$UV_CACHE_DIR \
    uv sync

ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
CMD ["uv", "run", "-m", "app.cli"]

VOLUME /app/storage
EXPOSE $APP_PORT
