from flask_jwt_extended import JWTManager
from flask_mail import Mail
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from flasgger import Swagger

from src.pixelface.api.config import Config

db = SQLAlchemy()
jwt = JWTManager()
swagger = Swagger()
mail = Mail()
socketio = SocketIO(cors_allowed_origins="*", message_queue=Config.SOCKETIO_MESSAGE_QUEUE)