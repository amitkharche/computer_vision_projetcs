# Use official Python slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set Streamlit environment
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLECORS=false

# Expose default Streamlit port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py"]
