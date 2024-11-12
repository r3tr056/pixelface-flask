import os

from flask import Flask, redirect, url_for
from flask_cors import CORS

import firebase_admin

from src.pixelface.api.extensions import (db, swagger, jwt, mail, socketio)
from src.pixelface.api.config import Config
from src.pixelface.api.routes.auth import auth_bp
from src.pixelface.api.routes.project import project_blueprint as project_bp
from src.pixelface.api.routes.video import video_blueprint as video_bp

def create_app():
    app = Flask('pixelface')
    app.config.from_object(Config)
    
    # initialize firebase before using firebase-related modules
    initialize_firebase()

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Initialize extensions with the app
    mail.init_app(app)
    db.init_app(app)
    jwt.init_app(app)
    swagger.init_app(app)
    socketio.init_app(app)

    # Initialize CORS
    CORS(app)

    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(project_bp, url_prefix='/project')
    app.register_blueprint(video_bp, url_prefix='/video')

    @app.route('/')
    def index():
        return redirect(url_for('swagger.ui'))
    
    return app

def initialize_firebase():
    if not firebase_admin._apps:
        cred = firebase_admin.credentials.Certificate('..\..\..\service_keys\pixelface-firebase-adminsdk.json')
        firebase_admin.initialize_app(cred, {'storageBucket': Config.FIREBASE_STORAGE_BUCKET})

if __name__ == '__main__':
    app = create_app()

    with app.app_context():
        db.create_all()
    
    app.run(debug=True, port=5000)