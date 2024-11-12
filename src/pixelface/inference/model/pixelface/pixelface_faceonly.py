import os
import cv2
import numpy as np
import torch

from basicsr.utils.download_util import load_file_from_url
from torchvision.transforms import functional as F

from inference.model.pixelface.vqvae import VQVAEGANMultiHeadTransformer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PixelFaceOnly():
    """
    Inference loader for PixelFace - Works only for the PixelFace CCTV Inference Pipeline

    Args:
        model_path (str) : The path to the RestoreFormer model.
        upscale (float) : The upscale of the final output (default=2)
        arch (str) : The RestoreFormer architecture. Options: RestoreFormer | RestoreFormer++
        device (torch.device) : The device to run the model on (CPU or GPU).
    """

    def __init__(self, model_path, upscale=2, device=None):
        self.upscale = upscale
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        # Initialize the RestoreFormer model (or architecture of your choice)
        self.RF = VQVAEGANMultiHeadTransformer(head_size=3, ex_multi_scale_num=1)  # Adjust as needed

        # Load the model weights
        if model_path.startswith('https://'):
            model_path = load_file_from_url(url=model_path, model_dir=os.path.join(ROOT_DIR, 'experiments/weights'), progress=True, filename=None)
        loadnet = torch.load(model_path)

        weights = loadnet['state_dict']
        new_weights = {}

        for k, v in weights.items():
            if k.startswith('vqvae.'):
                k = k.replace('vqvae.', '')
            new_weights[k] = v
        self.RF.load_state_dict(new_weights)

        self.RF.eval()
        self.RF.to(self.device)

    @torch.no_grad()
    def enhance(self, cropped_face):
        """ Restore and upscale the cropped face image.

        Args:
            cropped_face (numpy.ndarray): The cropped face image.

        Returns:
            restored_face (numpy.ndarray): The restored and upscaled face image.
        """
        # Resize the face to 512x512 for the model input
        cropped_face_resized = cv2.resize(cropped_face, (512, 512))

        # Convert to tensor and normalize
        cropped_face_t = F.to_tensor(cropped_face_resized).unsqueeze(0).to(self.device)
        cropped_face_t = (cropped_face_t - 0.5) / 0.5  # Normalize to [-1, 1]

        # Run inference
        try:
            output = self.RF(cropped_face_t)[0]
            restored_face = (output.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255  # Denormalize and convert to uint8
            restored_face = np.clip(restored_face, 0, 255).astype('uint8')  # Ensure pixel values are in range
        except RuntimeError as error:
            print(f'Failed inference for RestoreFormer: {error}.')
            restored_face = cropped_face  # Return the original face if an error occurs

        return restored_face
