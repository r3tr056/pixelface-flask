
import torch
import random
import time
import cv2
import logging
import numpy as np

from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment
from basicsr.utils import img2tensor, imwrite, tensor2img, imfrombytes
from basicsr.data.data_util import paths_from_folder


class FFHQDataset(data.Dataset):
	""" FFHQ Dataset for StyleGAN"""

	def __init__(self, opt):
		super().__init__()

		self.opt = opt
		self.file_client = None

		self.mean = opt['mean']
		self.std = opt['std']

		self.file_client = GoogelDriveFileClient()
		self.image_ids = self.get_image_ids_from_google_drive(opt['dataroot_gt'])

		self.grey_prob = opt.get('gray_prob')

		self.exposure_prob = opt.get('exposure_prob', 0.)
		self.exposure_range = opt['exposure_range']

		self.shift_prob = opt.get('shift_prob', 0.)
		self.shift_unit = opt.get('shift_unit', 32)
		self.shift_max_num = opt.get('shift_max_num', 3)

		logger = logging.getLogger(self.__class__.__name__)

	def __getitem__(self, index):
		file_id = self.image_ids[index]
		local_image_path = f"temp_image_{index}.png"

		retry = 3
		while retry > 0:
			try:
				# stream image from google drive
				self.file_client.get(file_id, local_image_path)
			except Exception as e:
				logger.warning(f'File client error: {e}, remaining retry items: {retry - 1}')
			else:
				break
			finally:
				retry -= 1

		img_gt = imfrombytes(cv2.imread(local_image_path), float32=True)

		# random horizontal flip
		img_gt = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False)
		h, w, _ = img_gt.shape

		# apply exposure (based on probablity)
		if (self.exposure_prob is not None) and (np.random.uniform() < self.exposure_prob):
			exp_scale = np.random.uniform(self.exposure_range[0], self.exposure_range[1])
			img_gt *= exp_scale

		# apply shift (based on probablity)
		if (self.shift_prob is not None) and (np.random.uniform() < self.shift_prob):
			shify_vertical_num = np.random.randint(0, self.shift_max_num * 2 + 1)
			shift_horizontal_num = np.random.randint(0, self.shift_max_num * 2 + 1)
			shift_v = self.shift_unit * shift_vertical_num
			shift_h = self.shift_unit * shift_horizontal_num
			img_gt_pad = np.pad(img_gt, ((self.shift_max_num * self.shift_unit, self.shift_max_num * self.shift_unit), (self.shift_max_num * self.shift_unit, self.shift_max_num * self.shift_unit), (0,0)), mode='symmetric')
			img_gt = img_gt_pad[shift_v:shift_v + h, shift_h: shift_h + w,:]

		if self.gray_prob and np.random.uniform() < self.gray_prob:
			img_gt = cv2.cvtcolor(img_gt, cv2.COLOR_BGR2GRAY)
			img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])

		img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
		# normalize
		normalize(img_gt, self.mean, self.std, inplace=True)
		# remove the local image
		os.remove(local_image_path)
		return {'gt': img_gt, 'gt_path': file_id}

	def __len__(self):
		return len(self.image_ids)

# Example test code ------------

import argparse
from omegaconf import OmegaConf
import pdb


if __name__ == "__main__":
	base = 'confgs/ROHQD.yaml'

	opt = OmegaConf.load(base)
	dataset = FFHQDataset(opt['data']['params']['train']['params'])

	for i in range(14):
		sample = dataset.getitem(i)