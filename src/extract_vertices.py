import bpy
import csv
import os
import random
from os.path import dirname, join


KEEP_PROB = 0.005
TARGET_TEXTURES = ["Female 01.png", ".HG_Eye_Color.png", "TeethTongueSet_C_2K.png", "TeethTongueSet_C_2K.png"]


def get_image_from_object(obj, target_name):
    mat = obj.active_material
    if not mat or not mat.use_nodes:
        return None

    clean_target = os.path.splitext(target_name)[0].lower()

    for node in mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE' and node.image:

            clean_name = os.path.splitext(node.image.name)[0].lower()

            if clean_name == clean_target:
                image = node.image

                image.reload()

                src_path = bpy.path.abspath(image.filepath)
                filename = os.path.basename(src_path)
                dst_path = os.path.join(output_dir, filename)
                image.save(filepath=dst_path)

                return image

    return None


def sample_image_pixel(img, u, v):
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))

    width, height = img.size
    x = int(u * (width  - 1))
    y = int(v * (height - 1))

    index = (y * width + x) * 4
    pixels = img.pixels

    r = pixels[index + 0]
    g = pixels[index + 1]
    b = pixels[index + 2]

    return r, g, b


root_name = "HG"

root_obj = bpy.data.objects.get(root_name)
if root_obj is None:
    raise ValueError(f"Could not find object: '{root_name}'")

if root_obj.type == 'MESH':
    mesh_objects = [root_obj]
else:
    mesh_objects = [child for child in root_obj.children_recursive if child.type == 'MESH']

depsgraph = bpy.context.evaluated_depsgraph_get()

output_dir = bpy.path.abspath("//dummy")
output_dir = join(dirname(dirname(output_dir)), "output")
os.makedirs(output_dir, exist_ok=True)

filepath = os.path.join(output_dir, "vertices.csv")

with open(filepath, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["object_name", "face_index", "vert_index", "x", "y", "z", "r", "g", "b"])

    for obj, target_texture in zip(mesh_objects, TARGET_TEXTURES):
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
        world_matrix = obj.matrix_world

        if not mesh.uv_layers:
            obj_eval.to_mesh_clear()
            continue

        uv_layer = mesh.uv_layers.active
        img = get_image_from_object(obj, target_texture)

        for poly in mesh.polygons:
            for loop_index in poly.loop_indices:
                if random.random() > KEEP_PROB:
                    continue
                loop = mesh.loops[loop_index]
                vert_index = loop.vertex_index
                v = mesh.vertices[vert_index]

                co_world = world_matrix @ v.co

                uv = uv_layer.data[loop_index].uv
                u, v_uv = uv.x, uv.y

                r, g, b = sample_image_pixel(img, u, v_uv)

                writer.writerow([
                    obj.name,
                    poly.index,
                    vert_index,
                    co_world.x,
                    co_world.y,
                    co_world.z,
                    r, g, b
                ])

        obj_eval.to_mesh_clear()

print(f"CSV saved: {filepath}")
