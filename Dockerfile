# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set environment variables to avoid building wheels
ENV PIP_NO_BUILD_ISOLATION=1
ENV PIP_PREFER_BINARY=1

WORKDIR /app

# Install only essential system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with pre-built wheels
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
