# Use a slim, stable base image for reproducibility
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Specify the default command to run when the container starts
# This will execute the entire simulation suite
CMD ["python", "runner.py"]