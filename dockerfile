# Use the official Python image from the Docker Hub
FROM python:3.11

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install virtualenv and create a virtual environment in the working directory
RUN python -m pip install --upgrade pip
RUN pip install virtualenv
RUN python -m venv /usr/src/app/venv

# Activate the virtual environment and install the dependencies
RUN /bin/bash -c "source /usr/src/app/venv/bin/activate && pip install --no-cache-dir -r requirements.txt"

# Copy the rest of the application code into the container
COPY . .

# Expose the port (if needed for other purposes)
EXPOSE 8501

# Default command to start the container
CMD ["/bin/bash", "-c", "source /usr/src/app/venv/bin/activate && exec bash"]