FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies needed for compiling numerical libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libeccodes-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install the Python package directly to prefix
RUN pip install --no-cache-dir --prefix=/install .

# --- Runtime Stage ---
FROM python:3.11-slim

WORKDIR /app

# Install runtime C libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeccodes0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy the actual scripts and code
COPY . .

# Default command for the pipeline
CMD ["python", "scripts/run_daily.py", "--demo"]
