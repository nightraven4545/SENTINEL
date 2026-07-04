FROM python:3.12-slim

WORKDIR /app

# Install torch from the CPU wheel index first — the default PyPI torch on
# Linux bundles CUDA (~2GB+); Sentinel's tiny autoencoder only needs CPU.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY .env.example .

# data/ is a volume so the DuckDB warehouse + price cache survive restarts
VOLUME /app/data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
