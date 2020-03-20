import bpy
import math
import pdb
import numpy as np
from mathutils import Vector
# to run in python console in blender
# f0 = "C:\GitHub\latest-python\Blender\1st curve import.py"
# exec(compile(open(f0).read(), f0, 'exec'))
# print all objects

def sin_bcoords():
    np.pi
    x = np.linspace(0, 4 * np.pi, 200)
    z = np.sin(x)
    y = 0 * x
    coords = []
    for i0, j0 in enumerate(x):
        c = [x[i0], y[i0], z[i0]]
        coords.append(c)
    return coords
print(np.round(coords, 2))
# sample data
coords = sin_bcoords()

# create the Curve Datablock
curveData = bpy.data.curves.new('myCurve', type='CURVE')
curveData.dimensions = '3D'
curveData.resolution_u = 2

# map coords to spline
polyline = curveData.splines.new('NURBS')
polyline.points.add(len(coords))
for i, coord in enumerate(coords):
    x, y, z = coord
    polyline.points[i].co = (x, y, z, 1)

# create Object
curveOB = bpy.data.objects.new('myCurve', curveData)
curveData.bevel_depth = 0.1
bpy.ops.curve.primitive_bezier_circle_add(enter_editmode=False,
                                          location=(0, 0, 0))
bpy.context.object.dimensions[0] = 0.05
bpy.context.object.dimensions[1] = 0.05
bpy.context.object.dimensions[2] = 0.05

bpy.data.objects.data.objects[
    'myCurve'].data.bevel_object = bpy.data.objects.data.objects[
    'BezierCircle']
# attach to scene and validate context
scn = bpy.context.collection
scn.objects.link(curveOB)
