import os
import cv2
import torch
from glob import glob
import numpy as np
from face_lib import face_lib

from inference.model.pixelface.facerestore import Pixel2Face, RestoreFormer

from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils import RealESRGANer
from torchvision import transforms, utils, models

from inference.model.prnet.prnet import PRN
from inference.utils import frontalize, write_obj_with_colors, write_obj_with_texture
from inference.utils.render import get_uv_mask, get_visibility

REALESR_MODEL_PATH = 'pt/realesr/RealESRGAN_x4plus.pth'
RESTOREFORMER_PLUSPLUS_MODEL_PATH = 'pt/restoreformer/last.ckpt'
PRN_MODEL_PATH = 'pt/prn/last.ckpt'

if not os.path.exists('pt/realesr/'):
    os.makedirs('pt/realesr/')
if not os.path.exists(REALESR_MODEL_PATH):
    os.system("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -O pt/realesr/RealESRGAN_x4plus.pth")
if not os.path.exists('pt/restoreformer/'):
    os.makedirs('pt/restoreformer/')
if not os.path.exists(RESTOREFORMER_PLUSPLUS_MODEL_PATH):
    os.system("wget https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer++.ckpt -O pt/restoreformer/last.ckpt")

class Pipeline:

    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path

        # face detection and annotated video output part
        self.face_detector = face_lib()
        
        self.pnet_model = PRN(PRN_MODEL_PATH)

        # face enhancement part RestoreFormer
        self.face_enhancer = None
        self.srvgg_net_compact = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        half = True if torch.cuda.is_available() else False
        self.upsampler = RealESRGANer(scale=4, model_path=REALESR_MODEL_PATH, model=self.model, tile=0, tile_pad=10, pre_pad=0, half=True)


        
    def process_video_with_motion_detection(self, video_path, output_path, skip_frames=5):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

        background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
        prev_faces = []
        detected_face_images = []

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # resize frame to speed up processing
            frame_resized = cv2.resize(frame, (640, 360))
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            # motion detection
            fg_mask = background_subtractor.apply(gray_frame)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = any(cv2.contourArea(cnt) > 500 for cnt in contours)

            if motion_detected and frame_count % skip_frames == 0:

                rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                no_of_faces, faces_coords = self.face_detector.faces_locations(frame_resized)

                if faces_coords and len(faces_coords) != len(prev_faces):
                    prev_faces = faces_coords
                    for idx, face in enumerate(faces_coords):
                        x, y, w, h = face
                        face_image = frame_resized[y:y+h, x:x+w]
                        face_image_path = os.path.join(output_path, f'face_{frame_count}_{idx}.png')
                        cv2.imwrite(face_image_path, face_image)
                        detected_face_images.append(face_image_path)

                        cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # generate face annotated video output
            if out is None:
                height, width, _ = frame_resized.shape
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

            out.write(frame_resized)
            frame_count += 1

        cap.release()
        if out:
            out.release()

        return detected_face_images, output_path
    
    def process_faces(self, face_images, scale=2, aligned='not_aligned'):
        for face in face_images:
            output, save_path = self.enhance_face(face, scale, aligned)

    def face_restoreformer(self, image, aligned='not_aligned', scale=2):
        if scale > 4:
            scale = 4
        try:
            extension = os.path.splitext(os.path.basename(str(img)))[1]
            img = cv2.imread(image, cv2.COLOR_BGR2RGB)

            h, w = img.shape[0:2]
            if h > 3500 or w > 3500:
                print('too large size')
                return None, None
            
            if h < 300:
                img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

            self.face_enhancer = RestoreFormer(model_path=RESTOREFORMER_PLUSPLUS_MODEL_PATH, upscale=2, arch='RestoreFormer++', bg_upsampler=self.upsampler)

            has_aligned = True if aligned == 'aligned' else False
            _, restored_aligned, restored_img = self.face_enhancer.enhance(img, has_aligned=has_aligned, only_center_face=False, paste_back=True)
            output = restored_aligned[0] if has_aligned else restored_img

            if scale != 2:
                interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)

            extension = 'png' if img.shape[2] == 4 else 'jpg'
            # take the save path
            save_path = f'output/out_{extension}'
            cv2.imwrite(save_path, output)

            return output, save_path
        except Exception as error:
            print('Exception :', error)
            return None, None
        
    def face_prnet(self, image_path, save_path, isFront, out_image, isTexture, texture_size):

        types = ('*jpg', '*png')
        image_path_list = []
        for files in types:
            image_path_list.extend(glob(os.path.join(image_path, files)))
        total_num = len(image_path_list)

        for i, image_path in enumerate(image_path_list):
            name = image_path.strip().split('/')[-1][:-4]

            image = cv2.imread(image_path)
            [h, w, c] = image.shape
            if c > 3:
                image = image[:, :, :3]

            if image.shape[0] == image.shape[1]:
                image = cv2.resize(image, (256, 256))
                pos = self.pnet_model.net_forward(image/255.)
            else:
                box = np.array([0, image.shape[1]-1, 0, image.shape[0]-1])
                pos = self.pnet_model.process(image, box)

            image = image / 255.
            if pos is None:
                continue

            vertices = self.pnet_model.get_vertices(pos)
            if isFront:
                save_vertices = frontalize(vertices)
            else:
                save_vertices = vertices.copy()
            save_vertices[:, 1] = h - 1 - save_vertices[:, 1]

            if out_image is not None:
                cv2.imwrite(out_image, image)

            colors = self.pnet_model.get_colors(image, vertices)
            if isTexture:
                if texture_size != 256:
                    pos_interpolated = cv2.resize(pos, (texture_size, texture_size), preserve_range=True)
                else:
                    pos_interpolated = pos.copy()
                texture = cv2.remap(image, pos_interpolated[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
                vertices_vis = get_visibility(vertices, self.pnet_model.triangles, h, w)
                uv_mask = get_uv_mask(vertices_vis, self.pnet_model.traiangles, self.pnet_model.uv_coords, h, w, self.pnet_model.resolution_op)
                uv_mask = cv2.resize(uv_mask, (texture_size, texture_size), preserve_range =True)
                texture = texture * uv_mask[:,:,np.newaxis]
                write_obj_with_texture(os.path.join(save_path, name + '.obj'), save_vertices, self.pnet_model.triangles, texture, self.pnet_model.uv_coords / self.pnet_model.resolution_op)
            else:
                write_obj_with_colors(os.path.join(save_path, name + '.obj'), save_vertices, self.pnet_model.triangles, colors)