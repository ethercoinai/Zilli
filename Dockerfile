FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY zilli/version.py zilli/version.py

RUN pip install --no-cache-dir -e ".[server]"

COPY . .

EXPOSE 8900

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import httpx; httpx.get('http://127.0.0.1:8900/healthz')"

ENTRYPOINT ["python3", "-m", "zilli.cli"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8900"]
