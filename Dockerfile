FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the essential files (your .dockerignore will protect you)
COPY main.py .
COPY engine.py .
COPY APPROACH.md .
COPY WALKTHROUGH.md .
COPY README.md .
COPY .python-version .
COPY railpack.json .
COPY railway.json .

# Copy the data directory WITH the correct files
COPY data/ ./data/

# Verify the files exist (optional - helps debugging)
RUN ls -la data/

# Port
EXPOSE 8000

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]