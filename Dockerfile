FROM python:3.11.5-bullseye

# Install dependencies
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y libmagic-dev poppler-utils tesseract-ocr libreoffice pandoc nginx supervisor
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Supervisor configurations
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Nginx configurations
COPY docker/default /etc/nginx/sites-enabled/default

# Copy the requirements.txt into the container at /app/requirements.txt
COPY requirements.txt requirements.txt

# Upgrade pip
RUN pip install --upgrade pip

# Install pip packages
RUN pip install -r requirements.txt --upgrade

# Copy the current directory contents into the container at /app
COPY .streamlit/  ./.streamlit/
COPY src/ ./src/

# Command to run supervisord
CMD ["/usr/bin/supervisord"]
