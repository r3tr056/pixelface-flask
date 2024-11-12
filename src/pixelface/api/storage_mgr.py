import hashlib
import threading
import os
import logging
import json
import shutil
import urllib.request
import time


def generate_file_hash(file):
    hasher = hashlib.sha256()
    file.seek(0)
    while chunk := file.read(8192):
        hasher.update(chunk)
    file.seek(0)
    return hasher.hexdigest()

class StorageManager:

	def __init__(self, base_upload_dir='uploads', log_dir='logs', model_dir='models'):
		self.base_upload_dir = base_upload_dir
		self.log_dir = log_dir
		self.model_dir = model_dir

		os.makedirs(self.base_upload_dir, exist_ok=True)
		os.makedirs(self.log_dir, exist_ok=True)
		os.makedirs(self.model_dir, exist_ok=True)

		download_thread = threading.Thread(target=self.download_model_weights)
		download_thread.start()

	def download_model_weights(self):
		restoreformer_model_dir = os.path.join(self.model_dir, 'restoreformer')
		restoreformer_model_path = os.path.join(restoreformer_model_dir, 'last.ckpt')

		os.makedirs(restoreformer_model_dir, exist_ok=True)
		
		if not os.path.exists(restoreformer_model_path):
			self.log_message("Downloading RestorFormer++ model...")
			url = "https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer++.ckpt"
			
			try:
				with urllib.request.urlopen(url) as response:
					file_size = int(response.getheader("Content-Length"))
					block_size = 8192
					downloaded = 0

					with open(restoreformer_model_path, "wb") as out_file:
						start_time = time.time()
						while True:
							buffer = response.read(block_size)
							if not buffer:
								break
							downloaded += len(buffer)
							out_file.write(buffer)

							# Calculate download progress
							percent = downloaded * 100 / file_size
							speed = downloaded / (time.time() - start_time) / 1024 / 1024  # in MB/s
							self.log_message(f"Download progress: {percent:.2f}% ({speed:.2f} MB/s)")
				self.log_message("RestoreFormer++ model download completed.")
				
			except Exception as e:
				self.log_message(f"Error downloading RestoreFormer++ model: {e}")

	def get_restoreformer_model_path(self):
		restoreformer_model_path = os.path.join(self.model_dir, 'restoreformer', 'last.ckpt')
		return restoreformer_model_path

	def create_project_dir(self, project_id):
		""" Create a directory for a new project """
		project_path = os.path.join(self.base_upload_dir, 'projects', project_id)
		os.makedirs(project_path, exist_ok=True)

		# Create subdirectories
		os.makedirs(os.path.join(project_path, 'video'), exist_ok=True)
		os.makedirs(os.path.join(project_path, 'faces'), exist_ok=True)
		os.makedirs(os.path.join(project_path, 'enhanced_faces'), exist_ok=True)
		os.makedirs(os.path.join(project_path, 'models'), exist_ok=True)
		os.makedirs(os.path.join(project_path, 'annotations'), exist_ok=True)

		return project_path
	
	def save_video(self, project_id, video_file, video_hash):
		"""Upload the video file to the project directory."""
		project_path = os.path.join(self.base_upload_dir, 'projects', project_id, 'video')
		video_path = os.path.join(project_path, f'{video_hash}.mp4')

		video_file.save(video_path)  # or use a function to handle uploads
		return video_path
	
	def save_face_image(self, project_id, face_image, face_index):
		"""Save a detected face image in the project directory."""
		faces_dir = os.path.join(self.base_upload_dir, 'projects', project_id, 'faces')
		face_image_path = os.path.join(faces_dir, f'face_{face_index}.jpg')

		face_image.save(face_image_path)  # Assuming face_image is a PIL Image
		return face_image_path
	
	def get_face_images_dir(self, project_id):
		faces_dir = os.path.join(self.base_upload_dir, 'projects', project_id, 'faces')
		return faces_dir
	
	def get_enhanced_face_path(self, project_id):
		enhanced_faces_dir = os.path.join(self.base_upload_dir, 'projects', project_id, 'enhanced_faces')
		return enhanced_faces_dir
	
	def get_processed_video_path(self, project_id):
		processed_video_path = os.path.dir(self.base_upload_dir, 'projects', project_id, 'processed_video')
		return processed_video_path

	def get_3d_model_path(self, project_id):
		models_dir = os.path.join(self.base_upload_dir, 'projects', project_id, 'models')
		return models_dir

	def save_annotations(self, project_id, annotations):
		"""Save face annotations in a JSON file."""
		annotations_dir = os.path.join(self.base_upload_dir, 'projects', project_id, 'annotations')
		annotations_path = os.path.join(annotations_dir, 'face_annotations.json')

		with open(annotations_path, 'w') as json_file:
			json.dump(annotations, json_file, indent=4)

		return annotations_path

	def cleanup_temp_files(self):
		"""Remove temporary files from the temp directory."""
		temp_dir = os.path.join(self.base_upload_dir, 'temp')
		if os.path.exists(temp_dir):
			shutil.rmtree(temp_dir)

	def create_temp_dir(self):
		"""Create a temporary directory for processing files."""
		temp_dir = os.path.join(self.base_upload_dir, 'temp')
		os.makedirs(temp_dir, exist_ok=True)
		return temp_dir

	def log_message(self, message):
		"""Log messages to a log file."""
		# log_file = os.path.join(self.log_dir, 'app.log')
		# with open(log_file, 'a') as f:
		# 	f.write(f"{message}\n")
		logging.log(message)


storage_manager = StorageManager()