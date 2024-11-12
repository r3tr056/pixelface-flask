from flask import request
from flask_jwt_extended import get_jwt_identity
from flask_socketio import join_room

from src.pixelface.api.app import socketio


@socketio.on('connect')
def handle_connect():
    current_user_email = get_jwt_identity()
    join_room(current_user_email)
    