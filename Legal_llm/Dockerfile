# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for mysqlclient
RUN apt-get update && \
    apt-get install -y gcc pkg-config libmariadb-dev-compat libmariadb-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask's default port
EXPOSE 5000

# Copy the rest of the application code
COPY . .

# Run the application
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]