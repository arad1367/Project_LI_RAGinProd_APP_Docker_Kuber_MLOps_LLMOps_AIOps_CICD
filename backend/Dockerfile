FROM python:3.12-slim

# Don't write .pyc files; flush logs immediately (good for container logging).
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app/backend

# Install dependencies FIRST so this layer is cached and only re-runs
# when requirements change — not every time you edit code.
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code.
COPY backend/ .

# Run as a non-root user (a security best practice — limits damage if the
# app is ever compromised).
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]