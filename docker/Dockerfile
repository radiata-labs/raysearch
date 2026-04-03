FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RAYSEARCH_API_HOST=0.0.0.0

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

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

RUN mkdir -p /app/config /app/.cache

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN uv pip install --system ".[service]" \
    && python -m playwright install chromium

EXPOSE 8000

CMD ["raysearch-api"]
