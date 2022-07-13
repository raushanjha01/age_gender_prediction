# Create the base image
FROM python:3.7-slim

# Change the working directory
WORKDIR /app/

# Install Dependency
COPY requirements.txt /app/
RUN pip install -r ./requirements.txt

# Copy local folder into the container
COPY app.py /app/
COPY age_model.pkl /app/
COPY ap_data_model.pkl /app/
COPY static/files/README.md /app/static/files/README.md
COPY templates/index.html /app/templates/index.html
COPY templates/download.html /app/templates/download.html

# Set "python" as the entry point
ENTRYPOINT ["python"]

# Set the command as the script name
CMD ["app.py"]

#Expose the post 8080.
EXPOSE 8080