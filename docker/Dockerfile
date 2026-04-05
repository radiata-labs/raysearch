# =============================================================================
# RaySearch API - Multi-stage Dockerfile
# =============================================================================
# Stage: deps - Install dependencies
# Stage: dev   - Development with hot reload
# Stage: prod  - Production with security hardening
# =============================================================================

# -----------------------------------------------------------------------------
# Stage: deps - Install system and Python dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

# Install system dependencies for Playwright (Chromium)
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libasound2 \
        libatk-bridge2.0-0 \
        libatk1.0-0 \
        libcups2 \
        libdbus-1-3 \
        libdrm2 \
        libgbm1 \
        libglib2.0-0 \
        libgtk-3-0 \
        libnspr4 \
        libnss3 \
        libx11-6 \
        libx11-xcb1 \
        libxcb1 \
        libxcomposite1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxkbcommon0 \
        libxrandr2 \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml README.md /app/

# Install Python dependencies and Playwright
RUN uv pip install --system ".[service]" \
    && python -m playwright install chromium \
    && python -m playwright install-deps chromium

# -----------------------------------------------------------------------------
# Stage: dev - Development with hot reload and debug tools
# -----------------------------------------------------------------------------
FROM deps AS dev

WORKDIR /app

# Copy source code
COPY src /app/src

# Create directories
RUN mkdir -p /app/config /app/.cache

# Development environment
ENV RAYSEARCH_API_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Use uvicorn with reload for development
CMD ["uvicorn", "raysearch.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# -----------------------------------------------------------------------------
# Stage: prod - Production with security hardening
# -----------------------------------------------------------------------------
FROM deps AS prod

WORKDIR /app

# Create non-root user for security
RUN groupadd --gid 1001 raysearch \
    && useradd --uid 1001 --gid raysearch --shell /bin/bash --create-home raysearch

# Copy source code
COPY src /app/src

# Create directories with proper ownership
RUN mkdir -p /app/config /app/.cache \
    && chown -R raysearch:raysearch /app

# Production environment
ENV RAYSEARCH_API_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV NODE_ENV=production

# Switch to non-root user
USER raysearch

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/healthz || exit 1

# Production command
CMD ["raysearch-api"]