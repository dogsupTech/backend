# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install pipenv
RUN pip install --upgrade pip && \
    pip install pipenv

# Install dependencies from Pipfile
RUN pipenv install --deploy --ignore-pipfile

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
CMD ["pipenv", "run", "python", "app.py"]
