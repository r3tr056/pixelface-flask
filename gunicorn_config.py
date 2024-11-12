import multiprocessing

# Server socket
bind = '0.0.0.0:5000'  # Address to bind to, change port as needed
backlog = 2048

# Workers and concurrency
workers = 1  # Number of Gunicorn workers (eventlet will handle concurrency)
worker_class = 'eventlet'  # Use eventlet for asynchronous Socket.IO
worker_connections = 1000  # Maximum number of simultaneous clients

# Threads
threads = 2  # Optional: Number of threads per worker (useful for some workloads)

# Logging
accesslog = '-'  # Log access to stdout
errorlog = '-'   # Log errors to stdout
loglevel = 'info'  # Log level (debug, info, warning, error, critical)

# Daemonize the Gunicorn process (if needed)
# daemon = True

# Process name
proc_name = 'flask_socketio_app'

# Number of processes
preload_app = True  # Load the app before the workers are forked to save memory
