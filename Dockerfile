# Use the official Python 3.10 image as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY ./requirements.txt /app/requirements.txt

# Upgrade pip and install the dependencies from requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade -r /app/requirements.txt


# Copy the src folder and models folder into the container
COPY ./src /app/src
# COPY ./models /app/models

# Expose port 8080 for the application
EXPOSE 5006

# Set the command to run the application
CMD ["panel", "serve", "/app/src/gui_of_person_detection_classifiers.py", "--address", "0.0.0.0", "--port", "5006", "--allow-websocket-origin", "*", "--num-procs", "2", "--num-threads", "0", "--index", "gui_of_person_detection_classifiers"]

RUN mkdir /.cache
RUN chmod 777 /.cache

