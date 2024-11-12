import os

import cv2
import torch

from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from src.pixelface.inference.model.pixelface.vqvae import VQVAEGANMultiHeadTransformer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PixelFace:
	""" Inference Loader for PixelFace (based on RestoreFormer)
	
	It will detect and crop faces, and then resize the face to 512x512.
	RestoreFormer is used to restored the resized faces
	The background is upsampled with the bg_upsampler
	Finally, the faces will be pasted back to the upsample background image

	Args:
		model_path (str) : The path to th GFPGAN model.
		upscale (float) : The upscale of the final output (default=2)
		arch (str) : The RestoreFormer arch. Option: RestoreFormer | RestoreFormer++
		bg_upsampler (nn.Module) : The upsampler for the background. Default = None
	"""

	def __init__(self, model_path, upscale=2, arch='RestoreFormer++', bg_upsampler=None, device=None):
		# upscaling ration
		self.upscale = upscale
		# background upsampling
		self.bg_upsampler = bg_upsampler

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

		# restore former model
		self.RF = VQVAEGANMultiHeadTransformer(head_size=3, ex_multi_scale_num=1)

		# FaceHelper handles face detection, alignment and image processing
		self.face_helper = FaceRestoreHelper(
			upscale,
			face_size=512,
			crop_ratio=(1, 1),
			det_model='retinaface_resnet50',
			save_ext='png',
			use_prase=True,
			device=self.device,
			model_rootpath=None
		)
		
		# donwload and load the model
		if model_path.startswith('https://'):
			model_path = load_file_from_url(url=model_path, model_dir=os.path.join(ROOT_DIR, 'experiments/weights'), progress=True, filename=None)
		loadnet = torch.load(model_path)

		strict = False
		weights = loadnet['state_dict']
		new_weights = {}

		# process the weights in correct format
		for k, v in weights.items():
			if k.startswith('vqvae.'):
				k = k.replace('vqvae.', '')
			new_weights[k] = v
		self.RF.load_state_dict(new_weights, strict=strict)
		# eval mode and move to device (CPU/GPU)
		self.RF.eval()
		self.RF = self.RF.to(self.device)

	@torch.no_grad()
	def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True):
		# clean all previous state
		self.face_helper.clean_all()

		
		if has_aligned:
			# if the face is aligned
			# resize to 512x512
			img = cv2.resize(img, (512, 512))
			# get the cropper faces
			self.face_helper.cropped_faces = [img]
		else:
			# if the face is not aligned, align the face in the image
			self.face_helper.read_image(img)
			# detect landmarks
			self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
			self.face_helper.align_wrap_face()

		# loop over all the cropped faces detected in the image
		for cropped_face in self.face_helper.cropped_faces:
			cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
			normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
			cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

			try:
				output = self.RF(cropped_face_t)[0]
				restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
			except RuntimeError as error:
				print(f'\tFailed inference for RestoreFormer: {error}.')
				restored_face = cropped_face

			restored_face = restored_face.astype('uint8')
			self.face_helper.add_restored_face(restored_face)

		if not has_aligned and paste_back:
			if self.bg_upsampler is not None:
				bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
			else:
				bg_img = None

			self.face_helper.get_inverse_affine(None)
			restored_image = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
			return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_image
		else:
			return self.face_helper.cropped_faces, self.face_helper.restored_faces, None
