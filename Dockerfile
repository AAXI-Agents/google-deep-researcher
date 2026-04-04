FROM python:3.12-slim

WORKDIR /app

# Copy dependency manifest first for better layer caching.
# Pin versions in pyproject.toml to ensure reproducible builds.
COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

EXPOSE 8000

# Use 1 worker per container; scale horizontally via orchestration.
# Override WORKERS to increase concurrency within a single container.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir src --workers ${WORKERS:-1}"]
