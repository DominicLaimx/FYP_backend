# Use a slim Python image
FROM python:3.11-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose port 8000 (standard for Azure)
EXPOSE 8000

# Start the app
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]