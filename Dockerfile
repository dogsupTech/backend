# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Install dependencies using Pipenv
RUN pipenv install --deploy --ignore-pipfile

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World

# Run app.py when the container launches using pipenv
CMD ["pipenv", "run", "python", "app.py"]
