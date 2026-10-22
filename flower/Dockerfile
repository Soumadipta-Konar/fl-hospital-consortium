FROM python:3.11-slim

WORKDIR /app

# Install git and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY flower/ /app/flower/

ENV PYTHONPATH=/app

CMD ["python", "flower/server.py"]
