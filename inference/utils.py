import subprocess
import numpy as np
import numpy as np
from skimage.io import imsave
import os
from utils.render import vis_of_vertices, render_texture
from scipy import ndimage


def convert_to_mp4(input_path):
	output_path = input_path.rsplit('.', 1)[0] + ".mp4"
	ffmpeg_command = ['ffmpeg', '-i', input_path, '-vcodec', 'libx264', '-acodec', 'aac', output_path]

	try:
		subprocess.run(ffmpeg_command, check=True)
	except subprocess.CalledProcessError as e:
		print(f"Error during conversion: {e}")
		return None

	return output_path
