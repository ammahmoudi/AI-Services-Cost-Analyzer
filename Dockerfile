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

# Initialize database and run application
CMD ["sh", "-c", "python -c 'from ai_cost_manager.database import init_db; init_db()' && python app.py"]
