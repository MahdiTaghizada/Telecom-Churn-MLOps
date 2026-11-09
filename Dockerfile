FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    TELECOM_CHURN_API_HOST=0.0.0.0 \
    TELECOM_CHURN_API_PORT=8000

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY models/champion ./models/champion

RUN pip install --no-cache-dir uv==0.9.17 \
    && uv sync --frozen --no-dev --no-install-project \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uv", "run", "--no-dev", "uvicorn", "telecom_churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
