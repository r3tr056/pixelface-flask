

class FFHQDegradationDataset(data.Dataset):
	def __init__(self, opt):
		super().__init__()

		self.opt = opt
		self.file_client = GoogelDriveFileClient()
		self.image_ids = self.get_image_ids_from_google_drive(opt['dataroot_gt'])

		self.mean = opt['mean']
		self.std = opt['std']
		self.out_size = opt['out_size']

		self.crop_components = opt.get('crop_components', False)
		self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1)

		if self.crop_components:
			self.components_list = torch.load(opt.get('component_path'))

		self.blur_kernel_size = opt['blur_kernel_size']
		self.kernel_list = opt['kernel_list']
		self.kernel_prob = opt['kernel_prob']
		self.blur_sigma = opt['blur_sigma']
		self.downsample_range = opt['downsample_range']
		self.noise_range = opt['noise_range']
		self.jpeg_range = opt['jpeg_range']

		# color jitter
		self.color_jitter_prob = opt.get('color_jitter_prob')
		self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob')
		self.color_jitter_shift = opt.get('color_jitter_shift', 20)

		# to gray
		self.gray_prob = opt.get('gray_prob')

		logger = logging.getLogger(self.__class__.__name__)
		logger.info(f'Blur: blur_kernel_size {self.blur_kernel_size}, ', f'sigma: [{", ".join(map(str, self.blur_sigma))}]')
		logger.info(f'Downsample: downsample_range [{", ".join(map(str, self.downsample_range))}]')
		logger.info(f'Noise: [{", ".join(map(str, self.noise_range))}]')
		logger.info(f'JPEG compression: [{", ".join(map(str, self.jpeg_range))}]')

		self.color_jitter_shift /= 255

	@staticmethod
	def color_jitter(img, shift):
		jitter_val = np.random.uniform(-shify, shift, 3).astype(np.float32)
		img = img + jitter_val
		img = np.clip(img, 0, 1)
		return img

	@staticmethod
	def color_jitter_pt(img, brightnetss, contrast, stauration, hue):
		fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def get_component_coordinates(self, index, status):
        components_bbox = self.components_list[f'{index:08d}']
        if status[0]:  # hflip
            # exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # modify the width coordinate
            components_bbox['left_eye'][0] = self.out_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.out_size - components_bbox['right_eye'][0]
            components_bbox['mouth'][0] = self.out_size - components_bbox['mouth'][0]

        # get coordinates
        locations = []
        for part in ['left_eye', 'right_eye', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations.append(loc)
        return locations

    def __getitem__(self, index):
    	self.file_client.get(file_id, local_image_path)
    	img_gt = imfrombytes(cv2.imread(local_image_path), float32=True)

    	# random horizontal flip
    	img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
    	h, w, _ = img_gt.shape

    	if self.crop_components:
    		locations = self.get_component_coordinates(index, status)
    		loc_lefy_eye, loc_right_eye, loc_mouth = locations

    	# blur
    	assert self.blur_kernel_size[0] < self.blur_kernel_size[1], "Wrong blur kernel size range"
    	cur_kernel_size = random.randint(self.blur_kernel_size[0], self.blur_kernel_size[1]) * 2 + 1
    	kernel = degredations.random_mixed_kernels(
    		self.kernel_list,
    		self.kernel_prob,
    		
    	)