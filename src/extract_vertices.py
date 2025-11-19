import bpy
import csv
import os
from os.path import dirname, join

obj_name = "Cube"
obj = bpy.data.objects.get(obj_name)

mesh = obj.data
world_matrix = obj.matrix_world

output_dir = bpy.path.abspath("//dummy")
output_dir = join(dirname(dirname(output_dir)), "output")
os.makedirs(output_dir, exist_ok=True)

filepath = os.path.join(output_dir, "vertices.csv")

with open(filepath, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["face_index", "vert_index", "x", "y", "z"])

    for poly in mesh.polygons:
        for vert_index in poly.vertices:
            v = mesh.vertices[vert_index]
            
            co_world = world_matrix @ v.co
            writer.writerow([
                poly.index,
                vert_index,
                co_world.x,
                co_world.y,
                co_world.z
            ])

print(f"CSV saved: {filepath}")
