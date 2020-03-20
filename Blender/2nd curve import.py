##############################################################################
# Import some libraries
##############################################################################
import bpy
import math
import pdb
import numpy as np
from mathutils import Vector

#               run these commands in python console in blender:
#               f0 = r"C:\GitHub\latest-python\Blender\2nd curve import.py"
#               exec(compile(open(f0).read(), f0, 'exec'))


##############################################################################
# Some defs
##############################################################################
def Gaussian_1D(x, A, x_c, σ, bkg=0, N=1):
    # Note the optional input N, used for super Gaussians (default = 1)
    x_c = float(x_c)
    G = A * np.exp(- (((x - x_c) ** 2) / (2 * σ ** 2))**N) + bkg
    return G


def Gauss_pulse(x, t):
    w = 1
    k = 1
    z = Gaussian_1D(x, 20, 0, 15) * np.real(np.exp(1j * (k * x - w * t)))
    y = 0 * x
    coords = []
    for i0, j0 in enumerate(x):
        c = [x[i0], y[i0], z[i0]]
        coords.append(c)
    return coords


def delete_string_obj(string):
    bpy.ops.object.select_all(action='DESELECT')
    for i0, v0 in enumerate(np.arange(0, len(bpy.context.scene.objects))):
        delete = string in bpy.context.scene.objects[v0].name
        name = bpy.context.scene.objects[v0].name
        if delete is True:
            print('deleting ', name)
            bpy.data.objects[name].select_set(True)
    bpy.ops.object.delete()


def create_curve(coords, name, location, width, material):
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
    curveOB = bpy.data.objects.new(name, curveData)
    curveData.bevel_depth = width
    bpy.ops.curve.primitive_bezier_circle_add(enter_editmode=False,
                                              location=(0, 0, 0))
    bpy.context.object.dimensions[0] = 0.1
    bpy.context.object.dimensions[1] = 0.1
    bpy.context.object.dimensions[2] = 0.1

    bpy.data.objects.data.objects[
        'Pulse'].data.bevel_object = bpy.data.objects.data.objects[
        'BezierCircle']

    # attach to scene and validate context
    scn = bpy.context.collection
    scn.objects.link(curveOB)
    bpy.context.view_layer.objects.active = bpy.context.scene.objects[-1]
    ob = bpy.context.active_object
    bpy.context.active_object.location.xyz = location

    # Get material
    mat = bpy.data.materials.get(material)
    if mat is None:
        # create material
        mat = bpy.data.materials.new(name="Material")

    # Assign it to object
    if ob.data.materials:
        # assign to 1st material slot
        ob.data.materials[0] = mat
    else:
        # no slots
        ob.data.materials.append(mat)

    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[curve].select_set(True)
    bpy.ops.object.editmode_toggle()
    bpy.ops.curve.de_select_last()
    bpy.ops.curve.delete(type='VERT')
    bpy.ops.object.editmode_toggle()
    return ob.name

##############################################################################
# Do some stuff
##############################################################################
pts = 150

# clear previous objects
delete_string_obj('BezierCircle')
delete_string_obj('Pulse')
delete_string_obj('Empty')


# create initial coords for pulse
x = np.linspace(-15 * np.pi, 15 * np.pi, pts)
t = 0
coords = Gauss_pulse(x, t)
offset = [0, 0, 0]
# create pulse
curve = create_curve(coords, 'Pulse', offset, 0.2, "z dep blue pulse")

# prepare for reploting
bpy.context.scene.frame_set(0)
bpy.ops.object.select_all(action='DESELECT')
bpy.data.objects[curve].select_set(True)

bpy.ops.object.editmode_toggle()
bpy.ops.curve.de_select_first()
bpy.ops.object.hook_add_newob()
bpy.ops.curve.select_next()
bpy.ops.curve.de_select_first()
bpy.ops.object.hook_add_newob()

for i1, v1 in enumerate(np.arange(0, pts - 2)):
    bpy.ops.curve.select_next()
    bpy.ops.curve.select_next()
    bpy.ops.curve.select_less()
    bpy.ops.object.hook_add_newob()


bpy.ops.object.editmode_toggle()
bpy.ops.object.select_all(action='DESELECT')

for i0, v0 in enumerate(np.arange(0, len(bpy.context.scene.objects))):
    hook = 'Empty' in bpy.context.scene.objects[v0].name
    name = bpy.context.scene.objects[v0].name
    if hook is True:
        bpy.data.objects[name].select_set(True)
        bpy.ops.anim.keyframe_insert_menu(type='Location')

ts = np.linspace(0, 2 * np.pi, 25)
p_coords = coords

for i0, v0 in enumerate(ts):
    bpy.context.scene.frame_set(i0)
    print(v0)
    coords = Gauss_pulse(x, v0)

    bpy.ops.object.select_all(action='DESELECT')
    i2 = 0
    for i1, v1 in enumerate(np.arange(0, len(bpy.context.scene.objects))):
        hook = 'Empty' in bpy.context.scene.objects[v1].name
        name = bpy.context.scene.objects[v1].name
        if hook is True:
            print(i2, name, coords[i2])
            bpy.data.objects[name].select_set(True)
            bpy.context.scene.objects[v1].location = coords[i2]
            bpy.ops.anim.keyframe_insert_menu(type='Location')
            i2 += 1
