# syntax=docker/dockerfile:1.7
# One image, several entrypoints: api | worker | train | promote (see docker/entrypoint.sh).
ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------------------
# build: install the locked dependencies and the package into /app/.venv with uv
# ---------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# Extra dependency groups, e.g. "--group train" for the trainer image (matplotlib).
ARG UV_GROUPS=""

# 1) Dependencies only. This (large: torch) layer is rebuilt only when pyproject.toml or
#    uv.lock change, not on every code change. --frozen: install exactly what the lock says.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-default-groups ${UV_GROUPS}

# 2) The project itself: small and fast to rebuild. --no-editable copies the package into
#    the venv, so the runtime stage does not need the source tree.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-editable ${UV_GROUPS}

# ---------------------------------------------------------------------------------------
# runtime: just Python + the venv. No uv, no build tools, non-root user.
# ---------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
ARG GIT_SHA=unknown
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GIT_SHA=${GIT_SHA} \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /app/data \
    && chown app:app /app /app/data
WORKDIR /app

COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app configs ./configs
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/entrypoint

USER app
EXPOSE 8000 9100
ENTRYPOINT ["entrypoint"]
CMD ["api"]
