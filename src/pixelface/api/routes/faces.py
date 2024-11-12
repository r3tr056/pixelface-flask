
from flask import Blueprint, jsonify
from flasgger import swag_from
from flask_jwt_extended import jwt_required, get_jwt_identity

from src.pixelface.api.models import Face, Project, User

face_blueprint = Blueprint('faces', __name__)

@face_blueprint.route('/<int:face_id>', methods=['GET'])
@jwt_required()
@swag_from({
    'summary': 'Get Face Details',
    'description': 'Retrieve details of a specific face by its face_id. User must be authorized to access the face.',
    'parameters': [
        {
            'name': 'face_id',
            'in': 'path',
            'type': 'integer',
            'required': True,
            'description': 'The ID of the face',
            'example': 1
        }
    ],
    'responses': {
        '200': {
            'description': 'Face details retrieved successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'face_id': {'type': 'integer'},
                    'image_path': {'type': 'string'},
                    'obj_file': {'type': 'string'}
                }
            }
        },
        '403': {
            'description': 'Unauthorized access',
            'schema': {'type': 'string'}
        },
        '404': {
            'description': 'Face not found',
            'schema': {'type': 'string'}
        }
    }
})
def get_face(face_id):
    face = Face.query.get_or_404(face_id)
    project = Project.query.get_or_404(id=face.project_id)
    user = User.query.get_or_404(id=project.user_id)

    current_user_email = get_jwt_identity()

    if user.email != current_user_email:
        return jsonify({'error': 'Unauthorized access'}), 403
    
    return jsonify({
        'face_id': face.id,
        'thumbnail_url': face.thumbnail_path,
        'texture_url': face.texture_path,
        'mtl_path': face.mtl_path,
        'obj_path': face.obj_path
    }), 200


@face_blueprint.route('/project/<int:project_id>', methods=['GET'])
@jwt_required()
@swag_from({
    'summary': 'List Faces in Project',
    'description': 'List all faces associated with a specific project. User must be authorized to access the project.',
    'parameters': [
        {
            'name': 'project_id',
            'in': 'path',
            'type': 'integer',
            'required': True,
            'description': 'The ID of the project',
            'example': 1
        }
    ],
    'responses': {
        '200': {
            'description': 'Faces listed successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'project_id': {'type': 'integer'},
                    'faces': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'face_id': {'type': 'integer'},
                                'image_path': {'type': 'string'},
                                'obj_file': {'type': 'string'}
                            }
                        }
                    }
                }
            }
        },
        '403': {
            'description': 'Unauthorized access',
            'schema': {'type': 'string'}
        },
        '404': {
            'description': 'Project or faces not found',
            'schema': {'type': 'string'}
        }
    }
})
def list_faces_for_project(project_id):
    current_user_email = get_jwt_identity()
    project = Project.query.get_or_404(id=project_id)
    user = User.query.get_or_404(id=project.user_id)

    if user.email != current_user_email:
        return jsonify({'error': 'Unauthorized access'}), 403
    
    faces = Face.query.filter_by(project_id=project.id).all()

    faces_list = [{
        'face_id': face.id,
        'thumbnail_url': face.thumbnail_path,
        'texture_url': face.texture_path,
        'mtl_path': face.mtl_path,
        'obj_path': face.obj_path
    } for face in faces]

    return jsonify({
        'project_id': project.id,
        'faces': faces_list
    }), 200