FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set the Python path to include your package
ENV PYTHONPATH=/app

# Change the CMD to use your actual entry point
CMD ["python", "lyrics_analyzer/app.py"]