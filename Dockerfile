# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Disable Conda and force pip-only environment
ENV PYTHONNOUSERSITE=1
ENV PATH="/usr/local/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip before installing dependencies
# RUN pip install --upgrade pip

# Copy the requirements file
COPY ./requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --prefer-binary --timeout 1200 -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app/

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]