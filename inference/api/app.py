from flask import Flask, render_template, Response, redirect, url_for, request, jsonify, send_file, session, g
from flask_uploads import UploadSet, configure_uploads, patch_request_class, ALL
from werkzeug.utils import secure_filename

import json
import logging
from logging.handlers import RotatingFileHandler
import threading
import time
import numpy as np
from random import random
import os
import sys
import cv2
import psutil

from face_lib import face_lib
from ..utils import convert_to_mp4


facesUpdateThread = threading.Thread()
monitoringThread = threading.Thread()

facesUpdateThread.daemon = False
monitoringThread.daemon = False

app = Flask('pixelface')
FL = face_lib()

app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOADED_VIDEOS_DEST'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

videos = UploadSet('videos', ALL)

configure_uploads(app, videos)
# 500 mb limit
patch_request_class(app, 500 * 1024 * 1024)

if not os.path.exists(app.config['UPLOADED_VIDEOS_DEST']):
	os.makedirs(app.config['UPLOADED_VIDEOS_DEST'])


def allowed_file(filename):
	return True

@app.route('/upload', methods=['POST'])
def upload_file():
	if 'file' not in request.files:
		return jsonify({"error": "No file part"}), 400

	file = request.files['file']

	if file.filename == '':
		return jsonify({"error": "No selected file"}), 400

	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file_url = videos.save(file, name=filename)
		return jsonify({"success": True, "message": "File uploaded", "filename": file_url}), 200

	return jsonify({"error": "Invalid file format"}), 400



def process_video(video_path, output_path):
	cap = cv2.VideoCapture(video_path)
	fourcc = cv2.VideoWriter(*'mp4v')
	out = None

	detected_face_images = []
	frame_count = 0

	while True:
		ret, frame = cap.read()
		if not ret:
			break
			
		frame_resized = cv2.resize(frame, (640, 360))
		no_of_faces, faces_coords = FL.faces_locations(frame_resized)

		if faces_coords:
			for idx, face in enumerate(faces_coords):
				ymin, xmin, ymax, xmax = face
				face_image = frame_resized[ymin:ymax, xmin:xmax]
				face_image_path = os.path.join(output_path, f'face_{frame_count}_{idx}.png')
				cv2.imwrite(face_image_path, face_image)
				detected_face_images.append(face_image_path)

				cv2.rectangle(frame_resized, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

		if out is None:
			height, width, _ = frame_resized.shape
			out_path = os.path.join(output_path, 'annotated_video.mp4')
			out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

	cap.release()
	if out:
		out.release()

	return detected_face_images, output_path


def process_faces(super_res_model, reconstruction_model):
	faces_dir = 'extracted_faces'
	face_images = [os.path.join(faces_dir, f) for f in os.listdir(faces_dir) if f.endswith('.png')]

	for face_image_path in face_images:
		face_image = cv2.imread(face_image)

def super_resolve_face(face_image, sr_model):
	pass

def reconstruct_3d_face(super_resolved_image, reconstruction_model):
	pass
