from celery import group

import jwt
from flask import Blueprint, request, jsonify
from flasgger import swag_from
from flask_socketio import join_room
from flask_jwt_extended import jwt_required, decode_token, get_jwt_identity

from src.pixelface.api.app import db, socketio
from src.pixelface.api.models import Project, User
from src.pixelface.api.storage_mgr import storage_manager, generate_file_hash
from src.worker.tasks import process_faces_task, video_annotation_task

video_blueprint = Blueprint('video', __name__)


@video_blueprint.route('/upload', methods=['POST'])
@jwt_required()
@swag_from({
    'summary': 'Upload Video for Processing',
    'description': 'Uploads a video file for face annotation and processing. Requires JWT authentication.',
    'parameters': [
        {
            'name': 'video',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'The video file to be uploaded',
        },
        {
            'name': 'project_id',
            'in': 'formData',
            'type': 'integer',
            'required': True,
            'description': 'The ID of the project to associate the video with',
            'example': 1
        }
    ],
    'responses': {
        '200': {
            'description': 'Video uploaded successfully and tasks initiated',
            'schema': {
                'type': 'object',
                'properties': {
                    'project_id': {'type': 'integer'},
                    'status': {'type': 'string'}
                }
            }
        },
        '403': {
            'description': 'Unauthorized access',
            'schema': {'type': 'string'}
        },
        '404': {
            'description': 'Project not found',
            'schema': {'type': 'string'}
        },
        '400': {
            'description': 'Required fields missing',
            'schema': {'type': 'string'}
        }
    }
})
def upload_video():
    try:
        current_user_email = get_jwt_identity()
        uploaded_video = request.files.get('video')
        project_id = request.form.get('project_id')

        if not uploaded_video or not project_id:
            return jsonify({'error': 'Required fields missing!'}), 400

        project = Project.query.filter_by(id=project_id).first()
        if not project:
            return jsonify({'message': 'Project not found!'}), 404
        
        user = User.query.filter_by(id=project.user_id).first()
        if user.email != current_user_email:
            return jsonify({'error': 'Unauthorized access'}), 403

        video_hash = generate_file_hash(uploaded_video)
        existing_project = Project.query.filter_by(video_hash=video_hash).first()
        if existing_project:
            return jsonify({
                'project_id': existing_project.project_id,
                'status': 'completed',
            }), 200
            
        video_path = storage_manager.save_video(project_id, uploaded_video, video_hash)
        project.video_hash = video_hash
        project.video_path = video_path
        project.status = 'uploading'
            
        db.session.commit()

        task = group(
            video_annotation_task.s(project.id, video_path, user_email=current_user_email),
            process_faces_task.s(project.id, user_email=current_user_email)
        ).apply_async()

        return jsonify({'project_id': project.id, 'status': 'uploaded', 'tid': task.id}), 200
    except Exception as ex:
        db.session.rollback()
        return jsonify({'error': str(ex)}), 500
    


@socketio.on('connect')
@jwt_required()
def handle_start_stream():
    token = request.args.get('token')
    if not token:
        return False
    try:
        decoded_token = decode_token(token)
        current_user_email = decoded_token['sub']
        user = User.query.filter_by(email=current_user_email).first()
        if not user: 
            socketio.emit('error', {'message': 'User not found!'})
            return
        
        user_room = f"{user.id}-room"
        join_room(user_room)
        socketio.emit('update', {
            'message': f"User {current_user_email} joined room {user_room}"
        }, room=user_room)
    except jwt.ExpiredSignatureError:
        print("Expired token")
        return False
    except jwt.InvalidTokenError:
        print("Invalid token")
        return False

@socketio.on('disconnect')
def handle_disconnect():
    print('User disconnected')