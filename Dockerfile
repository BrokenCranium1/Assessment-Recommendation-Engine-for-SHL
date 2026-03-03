# Use slim Python image (much smaller)
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ONLY what's needed (your .dockerignore will be used!)
COPY main.py .
COPY engine.py .
COPY APPROACH.md .
COPY WALKTHROUGH.md .
COPY README.md .
COPY test_predictions.csv .
COPY railpack.json .
COPY railway.json .
COPY .python-version .

# Create data directory and copy only final CSV
RUN mkdir -p data
COPY data/shl_catalog_final.csv ./data/
COPY data/test_predictions.csv ./data/

# Create scripts directory if needed
RUN mkdir -p scripts
COPY scripts/ ./scripts/

# Port
EXPOSE 8000

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]