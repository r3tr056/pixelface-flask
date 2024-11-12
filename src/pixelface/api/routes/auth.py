
from flask import (Blueprint, request, jsonify)
from flasgger import swag_from
from itsdangerous import URLSafeTimedSerializer
from flask_jwt_extended import (get_jwt, create_access_token, get_jwt_identity, jwt_required, create_refresh_token)
from firebase_admin import auth as firebase_auth
from flask_mail import Message

from src.pixelface.api.config import Config
from src.pixelface.api.models import User, TokenBlacklist
from src.pixelface.api.app import db

auth_bp = Blueprint('auth', __name__)
serializer = URLSafeTimedSerializer(Config.URL_SAFE_SECRET_KEY)

@auth_bp.route('/signin', methods=['POST'])
@swag_from({
    'summary': 'User Sign-in',
    'description': 'Log in with email and password to get an access token.',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'email': {'type': 'string', 'example': 'user@example.com'},
                    'password': {'type': 'string', 'example': 'password123'}
                },
                'required': ['email', 'password']
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Signin successful',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string', 'example': 'Signin successful'},
                    'access_token': {'type': 'string'}
                }
            }
        },
        '401': {
            'description': 'Invalid credentials',
            'schema': {'type': 'string'}
        }
    }
})
def signin():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        access_token = create_access_token(identity=user.email)
        refresh_token = create_refresh_token(identity=user.email)
        return jsonify({'message': 'Signin successful', 'access_token': access_token, 'refresh_token': refresh_token}), 200
    
    return jsonify({'message': 'Invalid credentials'}), 401


@auth_bp.route('/signup', methods=['POST'])
@swag_from({
    'summary': 'User Sign-up',
    'description': 'Register a new user with an email and password.',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'email': {'type': 'string', 'example': 'newuser@example.com'},
                    'password': {'type': 'string', 'example': 'password123'}
                },
                'required': ['email', 'password']
            }
        }
    ],
    'responses': {
        '201': {
            'description': 'Signup successful',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string', 'example': 'Signup successful'},
                    'user': {'type': 'string'}
                }
            }
        },
        '400': {
            'description': 'User already exists',
            'schema': {'type': 'string'}
        }
    }
})
def signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if User.query.filter_by(email=email).first():
        return jsonify({'message': 'User already exists'}), 400
    
    try:
        new_user = User(email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'message': 'Signup successful', 'user': email}), 201
    except Exception as ex:
        return jsonify({'message': 'Error occurred during signup', 'error': str(ex)}), 500


@auth_bp.route('/login/google', methods=['POST'])
@swag_from({
    'summary': 'Google Login',
    'description': 'Login using Google authentication token.',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'token': {'type': 'string', 'example': 'GoogleIDToken123'}
                },
                'required': ['token']
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Google login successful',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string', 'example': 'Google login successful'},
                    'access_token': {'type': 'string'}
                }
            }
        },
        '401': {
            'description': 'Invalid token',
            'schema': {'type': 'string'}
        }
    }
})
def google_login():
    data = request.json
    token = data.get('token')

    try:
        # verify the token with the firebase admin sdk
        decoded_token = firebase_auth.verify_id_token(token)
        email = decoded_token['email']

        user = User.query.filter_by(email=email).first()

        if user and not user.google_user:
            return jsonify({'message': 'This email is already registered with password, consider linking.'}), 409

        if not user:
            user = User(email=email, google_user=True)
            db.session.add(user)
            db.session.commit()

        access_token = create_access_token(identity=user.email)
        return jsonify({'message': 'Google login successful', 'access_token': access_token}), 200
    except Exception as ex:
        return jsonify({'message': 'Invalid token', 'error': str(ex)}), 401


@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
@swag_from({
    'summary': 'User Profile',
    'description': 'Get the profile of the logged-in user.',
    'responses': {
        '200': {
            'description': 'User profile data',
            'schema': {
                'type': 'object',
                'properties': {
                    'email': {'type': 'string', 'example': 'user@example.com'},
                    'google_user': {'type': 'boolean', 'example': True}
                }
            }
        },
        '404': {
            'description': 'User not found',
            'schema': {'type': 'string'}
        }
    }
})
def profile():
    current_user_email = get_jwt_identity()
    user = User.query.filter_by(email=current_user_email).first()
    if user:
        return jsonify({
            'email': user.email,
            'google_user': user.google_user,
        }), 200
    return jsonify({'message': 'User not found'}), 404

@auth_bp.route('/password-reset', methods=['POST'])
def password_reset_request():
    email = request.json.get('email')
    user = User.query.filter_by(email=email).first()

    if user:
        token = serializer.dump(user.email, salt='password_reset_salt')
        reset_url = f"{Config.FORNTEND_URL}/reset-password/{token}"
        msg = Message("Password Reset Request", sender=f"noreply@{Config.EMAIL_HANDLE}", recipients=[email])
        msg.body = f"Click the link to reset your password: {reset_url}"
        mail.send(msg)
        return jsonify({"message": "Password reset email sent!"}), 200
    return jsonify({'message': 'User not found'}), 404

@auth_bp.route('/reset-password/<token>', methods=['POST'])
def reset_password(token):
    try:
        email = serializer.loads(token, salt='password_reset_salt', max_age=3600)
        user = User.query.filter_by(email=email).first()
        if user:
            password = request.json.get('password')
            user.set_password(password)
            db.session.commit()
            return jsonify({'message': 'Password reset successful'}), 200
    except Exception as ex:
        return jsonify({'message': 'Invalid or expired token'}), 400
    
@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    current_user = get_jwt_identity()
    new_access_token = create_access_token(identity=current_user)
    return jsonify({'access_token': new_access_token}), 200

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    jti = get_jwt()['jti']
    try:
        token = TokenBlacklist(jti=jti)
        db.session.add(token)
        db.session.commit()
        return jsonify({'message': 'Successfully logged out'}), 200
    except Exception as ex:
        return jsonify({'message': 'Logout failed', 'error': str(ex)}), 500