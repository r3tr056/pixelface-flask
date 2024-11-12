

import numpy as np
import os
from glob import glob
from skimage.io import imread, imsave
from skimage.transform import resize, rescale
import scipy.io as sio
from inference.model.prnet.prnet import PRN

from utils.texture_utils import *
from utils.render import *
from inference.utils.obj_utils import *

def frontalize():
    pass

def generate_3d_model(
    input_dir,
    output_dir,
    is_dlib=True,
    is_texture=True,
    texture_size=256,
    is_frontalize=False,
    is_depth=False,
    is_mask=False,
    show_results=False,
    gpu='-1'
):
    """ Method to generate all files required for 3d model rendering in Three.JS
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output files (OBJ, MTL, texture_map)
        is_dlib: Wherether to use dlib for face detection
        is_texture: Whether to save texture information in the OBJ file
        texture_size: Size of the texture map (default: 256)
        is_frontalize : Whether to frontalize the vertices
        is_depth: Whether to output depth images
        is_mask: Whether to output invisible pixels in texture due to self-occlusion
        show_results: Whether to display the results (requires opencv)
        gpu: GPU id (set -1 for CPU)
    """

    

    prn = PRN(is_dlib=is_depth)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = []
    for ext in ('*.jpg', '*.png'):
        image_paths.extend(glob(os.path.join(input_dir, ext)))

    for image_path in image_paths:
        name = os.path.basename(image_path).split('.')[0]

        # read image
        image = imread(image_path)
        h, w, c = image.shape
        if c > 3:
            image = image[:, :, 3]

        if is_dlib:
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
        if is_frontalize:
            save_vertices = frontalize(vertices)
        else:
            save_vertices = vertices.copy()

        # adjust vertex coordinates
        save_vertices[:, 1] = h - 1 - save_vertices()

        # save textures or colors
        if is_texture:
            pos_interpolated = resize(pos, (texture_size, texture_size), preserve_range=True)
            texture = prn.get_texture(image, pos_interpolated)

            # apply mask for occlusions
            if is_mask:
                vertices_vis = get_visibility(vertices, prn.triangles, h, w)
                uv_mask = get_uv_mask(vertices_vis, prn.triangles)
                uv_mask = resize(uv_mask, (texture_size, texture_size), preserve_range=True)
                texture *= uv_mask[:, :, np.newaxis]

            write_mtl_textures(obj_name=name, vertices=save_vertices, triangles=prn.triangles, texture=texture, uv_coords=prn.uv_coords / prn.resolution_op, save_path=output_dir)

        else:
            colors = prn.get_colors(image, vertices)
            write_obj_with_colors(obj_name=name, vertices=save_vertices, triangles=prn.triangles, colors=colors, save_path=output_dir)
        
        if is_depth:
            depth = prn.get_depth_image(vertices, prn.triangles, h, w)
            imsave(os.path.join(output_dir, f'{name}_depth.jpg'), depth)
        