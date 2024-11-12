import os
from skimage.io import imsave


def write_mtl_textures(obj_name, vertices, triangles, texture, uv_coords, save_path):
    """
    Save #D face model with texture represented by texture map
    Modifier to generate MTL compatible with Three.js or Unity engine web.

    Args:
        obj_name: str - Name of the OBJ file to be saved
        vertices: shape - Shape (nver, 3), the 3D coordinates of vertices
        triangles: shape - Shape (ntri, 3), the indices of vertices forming triangles
        texture: shape - Shape (256,256,3), the texture map image.
        uv_coords: shape - Shape (nver, 3), the UV coordinates for texture mapping (max value<=1)
        save_path: str - Directory where the OBJ, MTL and texture files will be saved
    """
    os.makedirs(save_path, exist_ok=True)


    if not obj_name.endswith('.obj'):
        obj_name += '.obj'

    mtl_name = obj_name.replace('.obj', '.mtl')
    texture_name = obj_name.replace('.obj', '_texture.png')

    obj_path = os.path.join(save_path, 'obj', obj_name)
    mtl_path = os.path.join(save_path, 'mtl', mtl_name)
    texture_path = os.path.join(save_path, 'texture', texture_name)
    
    # ensure indices start at 1 (Meshlab)
    triangles = triangles.copy() + 1
    
    # write obj
    with open(obj_path, 'w') as f:
        # reference the mtl file using a relative path
        f.write("mtllib {}\n".format(mtl_name))

        # write vertices positions
        for i in range(vertices.shape[0]):
            f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        
        # Write UV texture Coordinates (flip V coordinates)
        for i in range(uv_coords.shape[0]):
            f.write('vt {} {}\n'.format(uv_coords[i,0], 1 - uv_coords[i,1]))

        # use the material defined in the MTL file
        f.write("usemtl FaceTexture\n")

        # Write face definitions with vertex and texture indices
        for i in range(triangles.shape[0]):
            f.write('f {}/{} {}/{} {}/{}\n'.format(triangles[i,2], triangles[i,2], triangles[i,1], triangles[i,1], triangles[i,0], triangles[i,0]))

    # Write MTL file
    with open(mtl_path, 'w') as f:
        f.write("newmtl FaceTexture\n")
        f.write("Kd 1.0 1.0 1.0\n") # Diffuse color (white)
        f.write("Ka 0.0 0.0 0.0\n")  # Ambient color
        f.write("Ks 0.0 0.0 0.0\n")  # Specular color
        f.write("d 1.0\n")           # Transparency
        f.write("Ns 0.0\n")          # Shininess
        # reference the texture image using a relative path
        s = 'map_Kd {}\n'.format(os.path.abspath(texture_name)) # map to image
        f.write(s)

    # Save the texture image
    imsave(texture_path, texture)


def write_obj_with_colors(obj_name, vertices, triangles, colors, save_path):
    """ Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
        save_path: str
    """
    os.makedirs(save_path, exist_ok=True)

    if not obj_name.endswith('.obj'):
        obj_name += '.obj'

    obj_filepath = os.path.join(save_path, 'obj', obj_name)
    # meshlab starts with 1
    triangles = triangles.copy() + 1
        
    # Write Vertices to the OBJ File
    with open(obj_filepath, 'w') as f:
        
        # write vertices & colors
        for i in range(vertices.shape[0]):
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)

        # Write face definitions (format: f v1 v2 v3)
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            # reordering the vertices to correct face orientation if needed
            s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
            f.write(s)




def write_obj_with_colors_texture(obj_name, vertices, colors, triangles, texture, uv_coords, save_path):
    """
    Save 3D face model with texture and colors.

    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
        texture: shape = (256,256,3)
        uv_coords: shape = (nver, 3) max value<=1
        save_path: str - Directory where OBJ, MTL, and texture files will be saved
    """

    os.makedirs(save_path, exist_ok=True)

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
    
    obj_filepath = os.path.join(save_path, 'obj', obj_name)
    mtl_name = obj_name.replace('.obj', '.mtl')
    mtl_filepath = os.path.join(save_path, 'mtl', mtl_name)
    texture_name = obj_name.replace('.obj', '_texture.png')
    texture_filepath = os.path.join(save_path, 'texture', texture_name)
    
    # Meshlab required indices start at 1
    triangles = triangles.copy() + 1
    
    # Write the OBJ file
    with open(obj_filepath, 'w') as f:
        # Write the reference to the material library (MTL file)
        s = "mtllib {}\n".format(os.path.abspath(mtl_name))
        f.write(s)

        # Write vertices and their corresponding colors
        for i in range(vertices.shape[0]):
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)
        
        # Write UV coordinates (for texture mapping)
        for i in range(uv_coords.shape[0]):
            s = 'vt {} {}\n'.format(uv_coords[i,0], 1 - uv_coords[i,1])
            f.write(s)

        # Use the material defined in the MTL file
        f.write("usemtl FaceTexture\n")

        # Write faces with vertices and UV coordinates (format: f v1/vt1 v2/vt2 v3/vt3)
        for i in range(triangles.shape[0]):
            s = 'f {}/{} {}/{} {}/{}\n'.format(triangles[i,2], triangles[i,2], triangles[i,1], triangles[i,1], triangles[i,0], triangles[i,0])
            f.write(s)

    # write mtl file
    with open(mtl_name, 'w') as f:
        f.write("newmtl FaceTexture\n")
        s = 'map_Kd {}\n'.format(os.path.abspath(texture_filepath)) # map to image
        f.write(s)

    # write texture as png
    imsave(texture_filepath, texture)
