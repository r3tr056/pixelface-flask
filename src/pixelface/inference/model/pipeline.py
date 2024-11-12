import os
import cv2
import logging
from glob import glob
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize, rescale

from src.pixelface.api.storage_mgr import StorageManager
from src.pixelface.inference.model.prnet.prnet import PRN
from src.pixelface.inference.model.pixelface.pixelface_faceonly import PixelFaceOnly
from src.pixelface.inference.utils.render import get_uv_mask, get_visibility
from src.pixelface.inference.utils.texture_utils import *
from src.pixelface.api.config import Config


class PixelFacePipeline:

	def __init__(self, project_id, storage_manager: StorageManager, faceupscaling=2):
		self.project_id = project_id
		self.storage_manager = storage_manager
		
		# prnet model
		self.pnet_model = PRN()
		self.pixelface_enhancer = PixelFaceOnly(
			model_path=storage_manager.get_restoreformer_model_path(),
			upscale=faceupscaling
		)

		self._warm_up()

	def _warm_up(self):
		# Create a dummy image tensor to trigger the pipeline's initial allocations
		pass

	def enhance_face(self, image_path):
		""" Super-resolve faces in the provided image

		Args:
			image_path(str) : Path of the input image file

		Returns:
			output(numpy.ndarray): The enhanced image
		"""
		scale = self.pixelface_enhancer.upscale
		scale = min(scale, 4)

		try:
			img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
			if img is None:
				logging.error(f"Failed to read image: {image_path}")
				return None, None
			
			h, w = img.shape[0:2]
			if h > 3500 or w > 3500:
				logging.error('Image size too large!')
				return None, None
			
			if h < 300:
				img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

			output = self.pixelface_enhancer.enhance(img)

			if scale != 2:
				interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
				output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
			return output
		except Exception as error:
			logging.error(f"Exception during face enhancement for {image_path}: {error}")
			return None
		
	def superscale_batch(self, input_path):
		""" Process all face images in the specified folder """
		output_path = self.storage_manager.get_enhanced_face_path(self.project_id, idx)
		image_files = glob(os.path.join(input_path, '*.[jJ][pP][gG]')) + glob(os.path.join(input_path, '*.[pP][nN][gG]'))
		enhanced_images = []
		
		for idx, image_path in enumerate(image_files):
			result = self.enhance_face(image_path)
			save_path = os.path.join(output_path, f'enhanced_face_{idx}.jpg')
			if save_path:
				cv2.imwrite(result, save_path)
				enhanced_images.append(save_path)

		return enhanced_images
	
	def generate_3d_model(self, input_images, save_path, texture_size=256):
		""" Method to generate all files required for 3d model using PRNet
		Args:
			input_dir: Directory containing input images
			texture_size: Size of the texture map (default: 256)
		"""
		prn = PRN()
		for image_path in input_images:
			try:
				name = os.path.basename(image_path).split('.')[0]
				image = imread(image_path)
				h, w, c = image.shape
				if c > 3:
					image = image[:, :, 3]

				max_size = max(h, w)
				if max_size > 1000:
					image = rescale(image, 1000. / max_size)
					image = (image * 255).astype(np.uint8)
					pos = prn.process(image)
				else:
					if h == w:
						image = resize(image, (256, 256))
						pos = prn.net_forward(image / 255.)
					else:
						box = np.array([0, w, -1, 0, h-1])
						pos = prn.process(image, box)

				if pos is None:
					continue

				vertices = prn.get_vertices(pos)
				save_vertices = vertices.copy()

				# adjust vertex coordinates
				save_vertices[:, 1] = h - 1 - save_vertices()

				pos_interpolated = resize(pos, (texture_size, texture_size), preserve_range=True)
				texture = prn.get_texture(image, pos_interpolated)

				vertices_vis = get_visibility(vertices, prn.triangles, h, w)
				uv_mask = get_uv_mask(vertices_vis, prn.triangles)
				uv_mask = resize(uv_mask, (texture_size, texture_size), preserve_range=True)
				texture *= uv_mask[:, :, np.newaxis]

				write_mtl_textures(obj_name=name, vertices=save_vertices, triangles=prn.triangles, texture=texture, uv_coords=prn.uv_coords / prn.resolution_op, save_path=save_path)
				
				depth = prn.get_depth_image(vertices, prn.triangles, h, w)
				imsave(os.path.join(save_path, f'{name}_depth.jpg'), depth)

				logging.info(f"3D model for {image_path} saved successfully.")
			except Exception as error:
				logging.error(f"Failed to generate 3D model for {image_path}: {error}")
				
	def run_pipeline(self):
		try:
			logging.info("Starting face enhancement process...")
			faces_dir = self.storage_manager.get_face_images_dir(self.project_id)
			enhances_images = self.superscale_batch(faces_dir=faces_dir)
			if not enhances_images:
				logging.error("No images were enhanced. Exiting pipeline.")
				return
			logging.info("Face enhancement completed. Proceeding to 3D model generation...")
			save_path = self.storage_manager.get_3d_model_path(project_id=self.project_id)
			self.generate_3d_model(enhances_images, save_path=save_path, texture_size=256)
			logging.info("3D model generation completed.")
		except Exception as e:
			logging.error(f"Pipeline failed: {e}")