import os
from typing import List
import cv2
import time
import base64
import logging
import numpy as np
import rollbar
from face_lib import face_lib

from celery import Celery
from celery.signals import task_failure
from flask_socketio import SocketIO

from src.pixelface.api.storage_mgr import storage_manager
from src.pixelface.api.utils import initialize_face_trackers, is_frontal_face, save_face_image, upload_3d_models, upload_processed_faces
from src.pixelface.inference.model.pipeline import PixelFacePipeline
from src.pixelface.api.models import Project, Face, User
from src.pixelface.api.app import db
from src.pixelface.api.config import Config

socketio = SocketIO(message_queue=Config.SOCKETIO_MESSAGE_QUEUE)

def celery_base_data_hook(request, data):
    data['framework'] = 'celery'

celery_app = Celery('processing_tasks', broker=Config.CELERY_BROKER_URL, backend=Config.CELERY_RESULT_BACKEND)
celery_app.config_from_object(Config)

rollbar.init(Config.ROLLBAR_KEY, Config.ROLLBAR_APP)
rollbar.BASE_DATA_HOOK = celery_base_data_hook


FL = face_lib()

MOTION_THRESHOLD = 1000

@task_failure.connect
def handle_task_failure(**kw):
    rollbar.report_exc_info(extra_data=kw)

@celery_app.task(bind=True, name='task.video_annotation_task')
def video_annotation_task(self, user_email, project_id, video_path):
    """
    Celery task of annotating faces in a video
    Parameters:
    - user_email - str, email of the user
    - project_id - str, ID of the project
    - video_path - str, path of the video file.
    """
    try:

        project = Project.query.get(project_id)
        project.status = 'streaming'
        db.session.commit()
        socketio.emit('task_update', {
            'status': 'video_annotation_task',
            'project_id': project_id,
            'progress': 0,
        })

        face_image_dir = storage_manager.get_face_images_dir(project_id)
        if not os.path.exists(face_image_dir):
            os.makedirs(face_image_dir)

        processed_video_path = storage_manager.get_processesd_video_path()

        cap = cv2.VideoCapture(video_path)

        total_frames, frame_height, frame_width, fps = get_video_properties(cap)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))

        unique_faces, trackers, tracker_initialized = [], [], False
        ret, prev_frame = preprocess_frame(cap.read()[1])

        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_processed = process_frame(cap, frame, prev_frame, unique_faces, face_image_dir, trackers, tracker_initialized)
            out.write(frame_processed)

            emit_progress('video_annotation_processing', project_id, frame_count, total_frames)

        complete_video_annotation(out, cap, project_id)
    except Exception as ex:
        handle_task_failure_with_logging(ex, project_id, 'video_annotation_failed')
        return 'video-annotation-task-failed'
    
    return 'video-annotation-task-completed'

@celery_app.task(bind=True, name='tasks.process_faces_task')
def process_faces_task(self, user_email, project_id):
    project = Project.query.get(project_id)
    user = User.query.filter_by(email=user_email)
    room_id = f"{user.id}-room"

    if not user:
        socketio.emit('task_update', {
            'status': 'error',
            'progress': 0,
            'status': 'User not found',
        }, room=room_id)

    socketio.emit('task_update', {
        'status': 'process_faces_started',
        'project_id': project_id,
        'progress': 0,
    }, room=room_id)

    pipeline = PixelFacePipeline(storage_manager=storage_manager)

    try:
        pipeline.run_pipeline()
    except Exception as ex:
        socketio.emit('task_update', {
            'project_id': project_id,
            'status': 'error',
            'message': str(ex)
        }, room=room_id)
        return 'face-processing-failed'

    public_urls = {}
    try:
        # upload faces and to firebase storage
        enhanced_faces_path = storage_manager.get_enhanced_face_path(project_id)
        public_urls['faces'] = upload_processed_faces(enhanced_faces_path, project_id)
        model_path = storage_manager.get_3d_model_path(project_id)
        public_urls['3dmodels'] = upload_3d_models(model_path, project_id)
    except Exception as e:
        logging.error(f"Failed to upload files to Firebase: {e}")
        socketio.emit('task_update', {
            'project_id': project_id,
            'status': 'upload_error',
            'message': str(e),
            'progress': 0,
        }, room=room_id)
        return 'face-processing-upload-failed'

    # Emit completion status with URLs
    socketio.emit('task_update', {
        'project_id': project_id,
        'status': 'process_faces_completed',
        'progress': 100,
        'urls': public_urls
    }, room=room_id)

    return 'face-processing-completed'

def update_project_status(project_id: int, status: str) -> Project:
    project = Project.query.get(project_id)
    project.status = status
    db.session.commit()
    return project

def get_video_properties(cap: cv2.VideoCapture) -> tuple:
    return (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FPS)),
    )

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """ Converts and blurs a frame to grayscale for motion detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.GaussianBlur(gray, (21,21), 0)

def process_frame(
    cap: cv2.VideoCapture,
    frame: np.ndarray,
    prev_frame: np.ndarray,
    unique_faces: List,
    face_image_dir: str,
    trackers: List,
    tracker_initialized: bool
) -> np.ndarray:
    """ Process each frame to detect and track faces """
    gray = preprocess_frame(frame)
    motion_detection = detect_motion(prev_frame, gray)

    if motion_detection or tracker_initialized:
        faces = FL.get_faces(frame)
        handle_detected_faces(faces, frame, unique_faces, face_image_dir, trackers)
    return frame

def detect_motion(prev_frame: np.ndarray, gray: np.ndarray) -> bool:
    """ Detects motion between frames. """
    frame_delta = cv2.absdiff(prev_frame, gray)
    thres = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    return np.sum(cv2.dilate(thres, None, iterations=2)) > MOTION_THRESHOLD

def is_unique_face(face: np.ndarray, unique_faces: List) -> bool:
    """ Checks if a face is unique based on embeddings"""
    face_encoding = FL.face_embeddings(face)
    return all(np.linalg.norm(existing_face - face_encoding) >= 6.0 for existing_face in unique_faces)

def handle_detected_faces(faces: List, frame: np.ndarray, unique_faces: List, face_image_dir: str, trackers: List):
    """Porcesses detected faces, saves images, and updates trackers"""
    if faces:
        for face in faces:
            if is_frontal_face(face) and is_unique_face(face, unique_faces):
                unique_faces.append(face)
                save_face_image(face, frame, face_image_dir)
                trackers.append(initialize_face_trackers(frame, face))

def emit_progress(status: str, project_id: int, frame_count: int, total_frames: int):
    """ Emits progress updates"""
    progress = int((frame_count / total_frames) * 100)
    socketio.emit('task_update', {
        'status': status,
        'project_id': project_id,
        'progress': progress
    })

def complete_video_annotation(out, cap, project_id: int):
    """ Completes the video annotation process and cleans up resources"""
    out.release()
    cap.release()
    emit_status('video_annotation_completed', project_id, 100)

def emit_status(status: str, project_id: int, progress: int, room: str = None):
    """ Emit status updates to the client """
    socketio.emit('task_update', {
        'status': status,
        'project_id': project_id,
        'progress': progress
    }, room=room)

def handle_task_failure_with_logging(ex: Exception, project_id: int, status: str):
    logging.error(f"Error in {status}: {ex}")
    socketio.emit('task_update', {
        'project_id': project_id,
        'status': status,
        'message': str(ex),
        'progress': 0
    })

if __name__ == '__main__':
    celery_app.worker_main()