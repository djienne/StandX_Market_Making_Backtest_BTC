# StandX Data Collector Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY standx_common.py .
COPY data_collector.py .
COPY read_parquet.py .
COPY test_data_collector.py .

# Create data directory
RUN mkdir -p /app/data

# Set timezone to UTC for consistent timestamps
ENV TZ=UTC

# Default command
CMD ["python", "data_collector.py"]
