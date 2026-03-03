FROM python:3.11-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files
COPY main.py .
COPY engine.py .
COPY *.md .
COPY .python-version .
COPY railpack.json .
COPY railway.json .

# Create data directory and copy ALL CSV files AND PKL files
RUN mkdir -p /app/data
COPY data/*.csv /app/data/
COPY data/*.pkl /app/data/   # ← ADD THIS LINE!

# Verify the files exist
RUN ls -la /app/data/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]