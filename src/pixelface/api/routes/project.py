from flask import Blueprint, request, jsonify
from flasgger import swag_from
from flask_jwt_extended import jwt_required, get_jwt_identity

from src.pixelface.api.app import db
from src.pixelface.api.models import Project, User
from src.pixelface.api.storage_mgr import storage_manager

project_blueprint = Blueprint('projects', __name__)

@project_blueprint.route('/create-project', methods=['POST'])
@jwt_required()
@swag_from({
    'summary': 'Create a New Project',
    'description': 'Creates a new project for the given user and initializes a project directory. Requires JWT authentication.',
    'parameters': [
        {
            'name': 'user_id',
            'in': 'json',
            'type': 'integer',
            'required': True,
            'description': 'The ID of the user who owns the project',
            'example': 1
        }
    ],
    'responses': {
        '200': {
            'description': 'Project created successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'project_id': {'type': 'integer'},
                    'status': {'type': 'string'}
                }
            }
        },
        '400': {
            'description': 'Invalid request or missing fields',
            'schema': {'type': 'string'}
        }
    }
})
def create_project():
    current_user_email = get_jwt_identity()
    user = User.query.filter_by(email=current_user_email).first_or_404()

    new_project = Project(user_id=user.id)
    db.session.add(new_project)
    db.session.commit()

    storage_manager.create_project_dir(new_project.id)

    return jsonify({'project_id': new_project.id, 'status': 'created'}), 200


@project_blueprint.route('/<int:project_id>', methods=['GET'])
@jwt_required()
@swag_from({
    'summary': 'Get Project Details',
    'description': 'Fetches the details of a specific project by its ID. Requires JWT authentication.',
    'parameters': [
        {
            'name': 'project_id',
            'in': 'path',
            'type': 'integer',
            'required': True,
            'description': 'The ID of the project to retrieve',
            'example': 1
        }
    ],
    'responses': {
        '200': {
            'description': 'Project details retrieved successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'project_id': {'type': 'integer'},
                    'video_path': {'type': 'string'},
                    'status': {'type': 'string'}
                }
            }
        },
        '404': {
            'description': 'Project not found',
            'schema': {'type': 'string'}
        }
    }
})
def get_project(project_id):
    current_user_email = get_jwt_identity()

    user = User.query.filter_by(email=current_user_email).first_or_404()
    project = Project.query.filter_by(id=project_id, user_id=user.id).first_or_404()

    return jsonify({
        'project_id': project.id,
        'streaming_video_path': project.video_path,
        'status': project.status,
    }), 200
