FROM python:3.10-alpine

LABEL maintainer="Ankur Debnath <dangerankur56@gmail.com>"

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY . /app
WORKDIR /app

RUN apt-get update && \
    apt-get install -y gcc && \
    apt-get install -y libpq-dev && \
    apt-get install -y libopencv-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 500

CMD ["gunicorn", "-c", "gunicorn_config.py", "server:app"]