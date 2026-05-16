# ── Tethys Greece — Dockerfile ──────────────────────────────────────────────
# HuggingFace Spaces: Docker SDK, port 7860
# Build: docker build -t Tethys-greece .
# Run:   docker run -p 7860:7860 \
#          -e CDS_KEY=your-key \
#          -e GROQ_KEY=your-key \
#          -v $(pwd)/era5_cache:/app/era5_cache \
#          Tethys-greece
# ────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# --- system deps (netCDF4 needs HDF5 + libcurl) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        libhdf5-dev \
        libnetcdf-dev \
        libcurl4-openssl-dev \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps (cached layer) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application files ---
COPY app.py server.py ui.html ./

# --- Cache directory (persists via mounted volume on HF Spaces) ---
RUN mkdir -p /app/era5_cache
ENV ERA5_CACHE_DIR=/app/era5_cache

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# API keys — override at runtime via Space secrets
# (never hard-code secrets in the image)
ENV PORT=7860

# Non-root user for security (HF Spaces best practice)
RUN useradd -m -u 1000 Tethys
RUN chown -R Tethys:Tethys /app
USER Tethys

CMD ["python", "app.py"]
