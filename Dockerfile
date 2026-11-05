FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc libxml2-dev libxslt1-dev && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONPATH=src
ENV PYTHONUNBUFFERED=1

# Command to run the bot in serve mode
CMD ["python", "-m", "techbot.main", "--serve"]
