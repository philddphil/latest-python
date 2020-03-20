import bpy  
from mathutils import Vector  

# weight  
w = 1 

# we don't have to use the Vector() notation.  
listOfVectors = [(0,0,0),(1,0,0),(2,0,0),(2,3,0),(0,2,1)]  
  
def MakePolyLine(objname, curvename, cList):  
    curvedata = bpy.data.curves.new(name=curvename, type='CURVE')  
    curvedata.dimensions = '3D'  
  
    objectdata = bpy.data.objects.new(objname, curvedata)  
    objectdata.location = (0,0,0) #object origin  
    bpy.context.collection.objects.link(objectdata)  
  
    polyline = curvedata.splines.new('NURBS')  
    polyline.points.add(len(cList)-1)  
    for num in range(len(cList)):  
        polyline.points[num].co = (cList[num])+(w,)  
  
    polyline.order_u = len(polyline.points)-1
    polyline.use_endpoint_u = True
    
  
MakePolyLine("NameOfMyCurveObject", "NameOfMyCurve", listOfVectors)

obj = bpy.data.objects['NameOfMyCurveObject']
obj.data.fill_mode = 'FULL'
obj.data.bevel_depth = 0.1
obj.data.bevel_resolution = 15  