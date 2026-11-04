# BawaGo AI Agent Prototype - Isolated Container
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose standard container / Hugging Face Spaces port
EXPOSE 7860
EXPOSE 8080

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PORT=7860
ENV HOST=0.0.0.0

# Start the standalone agent server
CMD ["python", "agent_server.py"]

