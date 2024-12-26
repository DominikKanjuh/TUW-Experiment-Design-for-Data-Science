FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories with proper permissions
RUN mkdir -p /root/.local/share/gpt-rec && \
    chmod -R 777 /root/.local/share

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variable to specify data directory
ENV XDG_DATA_HOME=/root/.local/share

ENTRYPOINT ["python", "script.py"]
