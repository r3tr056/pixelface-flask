import os

class Config:
    DOMAIN = os.getenv('DOMAIN')
    FORNTEND_URL = os.getenv('FORNTEND_URL')
    MESSAGE_BROKER = os.getenv('MESSAGE_BROKER')
    SOCKETIO_MESSAGE_QUEUE = f'{MESSAGE_BROKER}/0'

    # celery config
    CELERY_RESULT_BACKEND = f'{MESSAGE_BROKER}/0'
    CELERY_BROKER_URL = f'{MESSAGE_BROKER}/0'
    ROLLBAR_KEY = os.getenv('ROLLBAR_KEY')
    ROLLBAR_APP = os.getenv('ROLLBAR_APP')
    WORKER_CONCURRENCY = int(os.getenv('WOKER_CONCURRENCY', 4))
    TASK_ACKS_LATE = bool(os.getenv('TASK_ACKS_LATE', 'True'))
    WORKER_PREFETCH_MULTIPLIER = int(os.getenv('WORKER_PREFETCH_MULTIPLIER', 1))
    
    # flask config
    ZIPKIN_SERVER = os.getenv('ZIPKIN_SERVER', 'http://localhost:9411/api/v1/spans')
    # sql config
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///pixelface.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = bool(os.getenv('SQLALCHEMY_TRACK_MODIFICATIONS', False))
    SECRET_KEY = os.getenv('SECREY_KEY', 'pixelface-ankurdeb')
    SECURITY_PASSWORD_SALT = os.getenv('SECURITY_PASSWORD_SALT', 'password_salt')
    URL_SAFE_SECRET_KEY = os.getenv('URL_SAFE_SECRET_KEY', 'urlsafe-pixelface-r3tr0')
    # jwt config
    JWT_SECRET_KEY = os.getenv('JWT_SECRET', 'pixelface-secret-11251')
    JWT_ACCESS_TOKEN_EXPIRES = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES', 3600))
    JWT_REFRESH_TOKEN_EXPIRES = int(os.getenv('JWT_REFRESH_TOKEN_EXPIRES', 86400))

    # mail config
    MAIL_SERVER = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.getenv('MAIL_PORT', 587))
    MAIL_USE_TLS = bool(os.getenv('MAIL_USE_TLS', True))
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.getenv('MAIL_DEFAULT_SENDER', f"noreply@{DOMAIN}")
    
    SWAGGER = {
        'title': 'PixelFace API',
        'uiversion': 3,
    }

    FIREBASE_STORAGE_BUCKET = 'your-app-id.appspot.com'