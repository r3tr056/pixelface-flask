FROM python:3.10-alpine

COPY . /worker
WORKDIR /worker

RUN pip install --no-cache-dir -r requirements.txt

CMD ["celery", "-A", "tasks", "worker", "--loglevel=info", "--logfile=logs/celery.log"]