FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal build tools (kept small) - required for some binary wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only the prod requirements first to leverage Docker cache
COPY requirements-prod.txt ./

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-prod.txt

# Copy project
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Production command. Use a single worker here; for more throughput use gunicorn+uvicorn workers.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
