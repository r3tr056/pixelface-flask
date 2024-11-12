"""
Bazel Face Detection Pipeline for Fast Face Detection on Low-Resource Machines
"""

import numpy as np
import torch
import cv2

from bazelface import BlazeFace

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

front_net = BlazeFace().to(gpu)
back_net = BlazeFace(back_model=True).to(gpu)

# load weights
front_net.load_weights("weights/blazeface.pth")
front_net.load_anchors("weights/anchors.npy")

back_net.load_weights("weights/blazefaceback.pth")
back_net.load_anchors("weights/anchorsback.npy")

front_net.min_score_thres = 0.75
front_net.min_suppression_threshold = 0.3

def plot_detections_on_frame(frame, detections, with_keypoints=True):
	if isinstance(detections, torch.Tensor):
		detections = detections.cpu().numpy()

	if detections.ndim == 1:
		detections = np.expand_dims(detections, axis=0)

	print("Found %d faces" % detections.shape[0])

	for i in range(detections.shape[0]):
		ymin = int(detections[i, 0] * frame.shape[0])
		xmin = int(detections[i, 1] * frame.shape[1])
		ymax = int(detections[i, 2] * frame.shape[0])
		xmax = int(detections[i, 3] * frame.shape[1])

		cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

		if with_keypoints:
			for k in range(6):
				kp_x = int(detections[i, 4 + k*2] * frame.shape[1])
				kp_y = int(detections[i, 4 + k*2 + 1] * frame.shape[0])
				cv2.circle(frame, (kp_x, kp_y), radius=3, color=(255, 0, 0), thickness=-1)
				
		return frame


def detect_faces(img_path):
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	front_detections = front_net.predict_on_image(img)
	front_detections.shape

	plot_detections_on_frame(front_detections)
	