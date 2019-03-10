import bpy
import math
import pdb
from mathutils import Vector
# to run in python console in blender
# filename = this path
# exec(compile(open(filename).read(), filename, 'exec'))
# print all objects
for obj in bpy.data.objects:
    print(obj.name)
    if("Curve" in obj.name):
        print("found")
        bpy.data.scenes["Scene"].objects.unlink(obj)
        bpy.data.objects.remove(obj)

# for cur in bpy.data.curves:
#     print(cur.name)
#     bpy.data.curves.remove(cur)

# sample data
coords = [(1, 0, 1), (2, 0, 0), (3, 0, 1)]

# create the Curve Datablock
curveData = bpy.data.curves.new('myCurve', type='CURVE')
curveData.dimensions = '3D'
curveData.resolution_u = 2

# map coords to spline
polyline = curveData.splines.new('POLY')
polyline.points.add(len(coords))
for i, coord in enumerate(coords):
    x, y, z = coord
    polyline.points[i].co = (x, y, z, 1)

# create Object
curveOB = bpy.data.objects.new('myCurve', curveData)
curveData.bevel_depth = 0.01
bpy.ops.curve.primitive_bezier_circle_add(view_align=False,
                                          enter_editmode=False,
                                          location=(
                                              -1.09052,
                                              -1.63116,
                                              0.0890484))
bpy.context.object.dimensions[0] = 2
bpy.context.object.dimensions[1] = 2


# attach to scene and validate context
scn = bpy.context.scene
scn.objects.link(curveOB)
scn.objects.active = curveOB
