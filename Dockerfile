# AI Services Cost Analyzer - Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install gunicorn for production
RUN pip install gunicorn

# Install Playwright and Chromium
RUN playwright install chromium && \
    playwright install-deps chromium

# Copy application code
COPY . .

# Create cache directory
RUN mkdir -p cache

# Expose Flask port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Make entrypoint executable
RUN chmod +x /app/entrypoint.py

# Run migrations and start application
CMD ["python", "entrypoint.py"]
