import subprocess
import cv2
import os
import re
import boto3
from botocore.exceptions import NoCredentialsError

from src.pixelface.api.config import Config

AWS_ACCESS_KEY_ID = Config.S3_ACCESS_KEY_ID
AWS_SECRET_KEY = Config.S3_SECRET_KEY
AWS_REGION = Config.AWS_REGION
AWS_S3_BUCKET_NAME = Config.AWS_S3_BUCKET_NAME

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def convert_to_mp4(input_path):
    output_path = input_path.rsplit('.', 1)[0] + ".mp4"
    ffmpeg_command = ['ffmpeg', '-i', input_path, '-vcodec', 'libx264', '-acodec', 'aac', output_path]

    try:
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        return None

    return output_path


def is_frontal_face(face_img):
    h, w, _ = face_img.shape
    aspect_ratio = w / h
    return 0.8 < aspect_ratio < 1.2

def save_face_image(face_img, frame_idx, output_folder):
    filename = f"{output_folder}/face_{frame_idx}.png"
    cv2.imwrite(filename, face_img)
    print(f"Saved : {filename}")

def initialize_face_trackers(frame, faces):
    trackers = []
    for face in faces:
        tracker = cv2.TrackerKCF_create()
        bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, bbox)
        trackers.append(tracker)
    return trackers

def upload_to_s3(local_path, remote_path):
    try:
        s3_client.upload_file(local_path, AWS_S3_BUCKET_NAME, remote_path, ExtraArgs={'ACL': 'public-read'})
        public_url = f"https://{AWS_S3_BUCKET_NAME}.s3.amazonaws.com/{remote_path}"
        return public_url
    except FileNotFoundError:
        print(f"File {local_path} not found.")
    except NoCredentialsError:
        print("AWS credentials not available.")
    return None

def upload_processed_faces(path, project_id):
    public_urls = []
    
    for face_file in os.listdir(path):
        if face_file.endswith('.jpg'):
            local_face_path = os.path.join(path, face_file)
            remote_face_path = f"enhanced_faces/{project_id}/{face_file}"
            public_url = upload_to_s3(local_face_path, remote_face_path)
            public_urls.append({'face': face_file, 'url': public_url})

    return public_urls


def upload_3d_models(path, project_id):
        models_list = []
        
        model_pattern = re.compile(r'enhanced_face_(\d+)(\.obj|\.mtl|_texture\.png|_depth\.jpg)$')

        current_model_urls = {}

        for model_file in os.listdir(path):

                match = model_pattern.match(model_file)
                if match:
                        index = match.group(1)
                        file_extension = match.group(2)

                        local_model_path = os.path.join(path, model_file)
                        remote_model_path = f"models/{project_id}/{model_file}"

                        public_url = upload_to_s3(local_model_path, remote_model_path)

                        if index not in current_model_urls:
                                current_model_urls[index] = {}
                                
                        if file_extension == '.obj':
                                current_model_urls[index]['obj'] = public_url
                        elif file_extension == '.mtl':
                                current_model_urls[index]['mtl'] = public_url
                        elif file_extension == '_texture.png':
                                current_model_urls[index]['texture'] = public_url
                        elif file_extension == '_depth.jpg':
                                current_model_urls[index]['depth'] = public_url

        for index, urls in current_model_urls.items():
                if all(key in urls for key in ['obj', 'mtl', 'texture', 'depth']):
                        models_list.append(urls)

        return models_list