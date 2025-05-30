# Use official Python image (v3.11 slim)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose port 8080 (Render uses this by default)
EXPOSE 8080

# Start the Flask app
CMD ["python3", "app.py"]
