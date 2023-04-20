# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import trimesh
import pyrender
import numpy as np
from pyrender import OrthographicCamera,PerspectiveCamera,IntrinsicsCamera,\
    DirectionalLight, SpotLight, PointLight,\
                         MetallicRoughnessMaterial,\
                         Primitive, Mesh, Node, Scene,\
                         Viewer, OffscreenRenderer, RenderFlags
import sys
# fuze_trimesh = trimesh.load("/data/swfcode/VIBE/skate/model/"+"skateboard"+".off")
# mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
# scene = pyrender.Scene()
# scene.add(mesh)
# pyrender.Viewer(scene, use_raymond_lighting=True)
# cam2 = PerspectiveCamera(yfov=(1))
# cam2 = PerspectiveCamera(yfov=0.0001, aspectRatio=1.414)
# cam = PerspectiveCamera()
# cam2 = Camera(znear=0, zfar=10000000,name='a')
# cam2 = OrthographicCamera(xmag=10, ymag=10)
# cam_pose = np.array([
    # [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0.5],
    # [1.0, 0.0,           0.0,           0.0],
    # [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 0.4],
    # [0.0,  0.0,           0.0,          1.0]
# ])
# cam_pose = np.array([
    # [0.0,  0, 0, 0],
    # [0, 0.0,           0.0,           0.0],
    # [0.0,0, 0, 0.0],
    # [0.0,  0.0,           0.0,          0]
# ])
# scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))

# mesh_node = scene.add(mesh )
# cam_node = scene.add(cam2)
# r = OffscreenRenderer(viewport_width=640*2, viewport_height=480*2)
# color, depth = r.render(scene)
# import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(color)
# plt.show()

# r.delete()
# fuze_trimesh = trimesh.load('examples/models/fuze.obj')
fuze_trimesh = trimesh.load("/data/swfcode/VIBE/skate/model/"+"skateboard"+".off")
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
scene = pyrender.Scene()
scene.add(mesh)
# camera = pyrender.PerspectiveCamera(yfov=10000, aspectRatio=100)
camera = pyrender.IntrinsicsCamera(cx=0.001,cy=0.00100,fx=0.001, fy=0.00100)
s = np.sqrt(2)/2+1
camera_pose = np.array([
    [0.0, -s,   s,   0.3],
    [1.0,  0.0, 0.0, 0.0],
    [0.0,  s,   s,   0.35],
    [0.0,  0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                           innerConeAngle=np.pi/16.0,
                           outerConeAngle=np.pi/6.0)
scene.add(light, pose=camera_pose)
r = pyrender.OffscreenRenderer(400, 400)
color, depth = r.render(scene)
plt.figure()
plt.subplot(1,2,1)
plt.axis('off')
plt.imshow(color)
plt.subplot(1,2,2)
plt.axis('off')
plt.imshow(depth, cmap=plt.cm.gray_r)
plt.show()
