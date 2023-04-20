# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import math
import trimesh
import pyrender
import numpy as np
from pyrender.constants import RenderFlags
from lib.models.smpl import get_smpl_faces


import fcl
def print_distance_result(o1_name, o2_name, result):
    print ('Distance between {} and {}:'.format(o1_name, o2_name))
    print ('-'*30)
    print ('Distance: {}'.format(result.min_distance))
    print ('Closest Points:')
    print (result.nearest_points[0])
    print (result.nearest_points[1])
    print ('')

def print_continuous_collision_result(o1_name, o2_name, result):
    print ('Continuous collision between {} and {}:'.format(o1_name, o2_name))
    print ('-'*30)
    print ('Collision?: {}'.format(result.is_collide))
    print ('Time of collision: {}'.format(result.time_of_contact))
    print ('')

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer:
    def __init__(self, resolution=(224,224), orig_img=False, wireframe=False):
        self.resolution = resolution

        self.faces = get_smpl_faces()
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

    def render(self, frame_num, img, verts, cam, angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9]):
        # print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)
        # mesh = trimesh.load("/data/swfcode/VIBE/horse/horse-gallop/horse-gallop-01.obj")

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)
        scale = [1, 1, 1]
        translate = [0.4,0.4,0]
        st=trimesh.transformations.scale_and_translate(scale, translate)
        mesh.apply_transform(st)

        delay =2
        step=int(frame_num%delay==0)
        num = (int(frame_num/delay)+step)%48+1

        # num=str("{:0>2d}".format(num))
        # print ("/data/swfcode/VIBE/horse/horse-gallop/horse-gallop-{:0>2d}.obj".format(num))
        # print ("/data/swfcode/VIBE/horse_animation/result/"+str(num)+".off")

        # mesh_horse = trimesh.load("/data/swfcode/VIBE/horse/horse-gallop/horse-gallop-{:0>2d}.obj".format(num))
        # mesh_horse = trimesh.load("/data/swfcode/VIBE/horse_animation/result/"+str(num)+".off")
        # mesh_horse = trimesh.load("/data/swfcode/VIBE/chairs/chair/"+"model-12"+".obj")
        # mesh_horse = as_mesh(mesh_horse)

        # Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        # Rx = trimesh.transformations.rotation_matrix(math.radians(30), [0, 1, 0.2])
        # mesh_horse.apply_transform(Rx)
        # animation='chair'
        # if animation=='horse':
            # scale = [2.0, 2.0, 2.0]
            # translate = [0.5,-1.2,-0.2]
        # elif animation=='chair':
            # scale = [2.0, 2.0, 2.0]
            # translate = [0.5,0,0.5]

        # st=trimesh.transformations.scale_and_translate(scale, translate)
        # mesh_horse.apply_transform(st)

        # if mesh_filename is not None:
            # print (mesh_filename)
            # mesh_horse.export(mesh_filename)

        # if angle and axis:
            # R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            # mesh_horse.apply_transform(R)
        # if mesh_filename is not None:
            # mesh_horse.export(mesh_filename)

        # if angle and axis:
            # R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            # mesh_horse.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000.
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        # mesh_horse = pyrender.Mesh.from_trimesh(mesh_horse, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')
        # mesh_horse_node = self.scene.add(mesh_horse, 'mesh_horse')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        # self.scene.remove_node(mesh_horse_node)
        self.scene.remove_node(cam_node)
        # print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        return image
    def render_objects(self, frame_num, img, verts, cam, angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9]):

        scene = pyrender.Scene()
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)
        # mesh = trimesh.load("/data/swfcode/VIBE/horse/horse-gallop/horse-gallop-01.obj")

        Rx = trimesh.transformations.rotation_matrix(math.radians(0), [1, 0, 0])
        mesh.apply_transform(Rx)
        scale = [1, 1, 1]
        translate = [0,0,0]
        st=trimesh.transformations.scale_and_translate(scale, translate)
        mesh.apply_transform(st)
        # # scene.add(mesh)

        # delay =2
        # step=int(frame_num%delay==0)
        # num = (int(frame_num/delay)+step)%300+1

        # num=str("{:0>2d}".format(num))
        # print ("/data/swfcode/VIBE/horse/horse-gallop/horse-gallop-{:0>2d}.obj".format(num))
        # print ("/data/swfcode/VIBE/horse_animation/result/"+str(num)+".off")

        # mesh_horse = trimesh.load("/data/swfcode/VIBE/horse/horse-gallop/horse-gallop-{:0>2d}.obj".format(num))
        # mesh_horse = trimesh.load("/data/swfcode/VIBE/horse_animation/result/"+str(num)+".off")
        mesh_horse = trimesh.load("/data/swfcode/VIBE/chairs/chair/"+"model-76"+".obj")#twopersons
        # mesh_horse = trimesh.load("/data/swfcode/VIBE/chairs/chair/"+"model-16"+".obj")#man
        mesh_horse = as_mesh(mesh_horse)

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [0, 1, 0])
        # Rx = trimesh.transformations.rotation_matrix(math.radians(30), [0, 1, 0.2])
        mesh_horse.apply_transform(Rx)
        mesh_scene = trimesh.Scene([mesh,mesh_horse])
        manager, objects = trimesh.collision.scene_to_collision(mesh_scene)
        # print ("============>", manager,objects,manager.in_collision_internal())
        req = fcl.DistanceRequest(enable_nearest_points=True)
        # print ("---------------")
        res = fcl.DistanceResult()
        # print ("---------------")
        # Create mesh geometry
        verts=mesh.vertices
        faces=mesh.faces
        mesh_bvh = fcl.BVHModel()
        mesh_bvh.beginModel(len(verts), len(faces))
        mesh_bvh.addSubModel(verts, faces)
        mesh_bvh.endModel()

        fcl1=fcl.CollisionObject(mesh_bvh, fcl.Transform())
        vertsh=mesh_horse.vertices
        facesh=mesh_horse.faces
        mesh_horse_bvh = fcl.BVHModel()
        mesh_horse_bvh.beginModel(len(vertsh), len(facesh))
        mesh_horse_bvh.addSubModel(vertsh, facesh)
        mesh_horse_bvh.endModel()
        # print ("---------------")
        fcl2=fcl.CollisionObject(mesh_horse_bvh, fcl.Transform(np.array([1.01,0.0,0.0])))
        # print ("---------------")
        dist = fcl.distance(fcl.CollisionObject(mesh_bvh, fcl.Transform()),
                                                fcl.CollisionObject(mesh_horse_bvh, fcl.Transform()),
                                                req, res)
        res_dist=res.nearest_points[1]-res.nearest_points[0]
        # animation='chair'
        # if animation=='horse':
            # scale = [2.0, 2.0, 2.0]
            # translate = [0.5,-1.2,-0.2]
        # elif animation=='chair':
            # scale = [1.5, 1.5, 1.5]
            # translate = [1,-0.3,-0.3]+res_dist
        print_distance_result('Box', 'Box', res)

        animation='chair'
        if animation=='horse':
            scale = [2.0, 2.0, 2.0]
            translate = [0.5,-1.2,-0.2]
        elif animation=='chair':
            scale = [1.5, 1.5, 1.5]
            # translate = [0.5,0.1,0.1]
            translate = [1,-0.3,-0.3]
            # translate = [0,0,0]

        st=trimesh.transformations.scale_and_translate(scale, translate)
        mesh_horse.apply_transform(st)
        # req = fcl.ContinuousCollisionRequest()
        # res = fcl.ContinuousCollisionResult()

        # dist = fcl.continuousCollide(fcl.CollisionObject(mesh_bvh, fcl.Transform()),
        #                                                           fcl.Transform(np.array([0.0, 0.0, 0.0])),
        #                                                           fcl.CollisionObject(mesh_horse_bvh, fcl.Transform()),
        #                                                           fcl.Transform(np.array([0.0, 0.0, 0.0])),
        #                                                           req, res)
        # print_continuous_collision_result('Box', 'Cylinder', res)
#
        if mesh_filename is not None:
            # print (mesh_filename)
            mesh_horse.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh_horse.apply_transform(R)
        if mesh_filename is not None:
            mesh_horse.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh_horse.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000.
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        # print("-- Distance result: ", dis)
        # print(result.nearest_points)

        # while (not manager.in_collision_internal()):
            # manager, objects = trimesh.collision.scene_to_collision(mesh_scene)
            # print ("============>", manager,objects,manager.in_collision_internal())
            # scale = [1.5, 1.5, 1.5]
            # translate = [-0.5+0.1*10,-0.5+0.1*10,-0.5+0.1*10]
            # for i in range(20,1):
                # for j in range(20,1):
                    # for k in range(20,1):
                        # translate = [-0.5+0.1*i,-0.5+0.1*j,-0.5+0.1*k]
                        # st=trimesh.transformations.scale_and_translate(scale, translate)
                        # mesh_horse.apply_transform(st)
                        # manager, objects = trimesh.collision.scene_to_collision(mesh_scene)
                        # if ( manager.in_collision_internal()):
                            # break
                    # else:
                        # continue
                    # break
                # else:
                    # continue
                # break
                        # print ("============>", manager,objects,manager.in_collision_internal())

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        mesh_horse = pyrender.Mesh.from_trimesh(mesh_horse, material=material)

        # mesh_node = self.scene.add(mesh, 'mesh')
        mesh_horse_node = self.scene.add(mesh_horse, 'mesh_horse')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :,:-1] * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)



        # self.scene.remove_node(mesh_node)
        self.scene.remove_node(mesh_horse_node)
        self.scene.remove_node(cam_node)
        # print (image)

        return image
    def render_objects_horse(self, frame_num, img, verts, cam, angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9]):

        scene = pyrender.Scene()
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)
        # mesh = trimesh.load("/data/swfcode/VIBE/horse/horse-gallop/horse-gallop-01.obj")

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)
        scale = [1.1, 1.1, 1.1]
        # translate = [0.5,0.5,-0.3]
        translate = [0.3,-0.1,0.2]
        st=trimesh.transformations.scale_and_translate(scale, translate)
        mesh.apply_transform(st)
        # # scene.add(mesh)

        delay =1
        step=int(frame_num%delay==0)
        num = (int(frame_num/delay)+step)%48+1

        # num=str("{:0>2d}".format(num))
        # print ("/data/swfcode/VIBE/horse/horse-gallop/horse-gallop-{:0>2d}.obj".format(num))
        # print ("/data/swfcode/VIBE/horse_animation/result/"+str(num)+".off")

        mesh_horse = trimesh.load("/data/swfcode/VIBE/horse/horse-gallop/horse-gallop-{:0>2d}.obj".format(num))
        # mesh_horse = trimesh.load("/data/swfcode/VIBE/horse_animation/result/"+str(num)+".off")
        # mesh_horse = trimesh.load("/data/swfcode/VIBE/chairs/chair/"+"model-76"+".obj")#twopersons
        # mesh_horse = trimesh.load("/data/swfcode/VIBE/chairs/chair/"+"model-16"+".obj")#man
        cate="horse"
        if cate=="chair":
            mesh_horse = as_mesh(mesh_horse)

        # Rx = trimesh.transformations.rotation_matrix(math.radians(180), [0, 1, 0])
        Rx = trimesh.transformations.rotation_matrix(math.radians(0), [0, 1, 0])
        mesh.apply_transform(Rx)
        #########################rotation for horse#####################3
        Rx = trimesh.transformations.rotation_matrix(math.radians(50), [0, 1, 0])
        mesh_horse.apply_transform(Rx)
        mesh_scene = trimesh.Scene([mesh,mesh_horse])
        manager, objects = trimesh.collision.scene_to_collision(mesh_scene)
        # print ("============>", manager,objects,manager.in_collision_internal())
        req = fcl.DistanceRequest(enable_nearest_points=True)
        # print ("---------------")
        res = fcl.DistanceResult()
        # print ("---------------")
        # Create mesh geometry
        verts=mesh.vertices
        faces=mesh.faces
        mesh_bvh = fcl.BVHModel()
        mesh_bvh.beginModel(len(verts), len(faces))
        mesh_bvh.addSubModel(verts, faces)
        mesh_bvh.endModel()

        fcl1=fcl.CollisionObject(mesh_bvh, fcl.Transform())
        vertsh=mesh_horse.vertices
        facesh=mesh_horse.faces
        mesh_horse_bvh = fcl.BVHModel()
        mesh_horse_bvh.beginModel(len(vertsh), len(facesh))
        mesh_horse_bvh.addSubModel(vertsh, facesh)
        mesh_horse_bvh.endModel()
        # print ("---------------")
        fcl2=fcl.CollisionObject(mesh_horse_bvh, fcl.Transform(np.array([1.01,0.0,0.0])))
        # print ("---------------")
        dist = fcl.distance(fcl.CollisionObject(mesh_bvh, fcl.Transform()),
                                                fcl.CollisionObject(mesh_horse_bvh, fcl.Transform()),
                                                req, res)
        res_dist=res.nearest_points[1]-res.nearest_points[0]
        # animation='chair'
        # if animation=='horse':
            # scale = [2.0, 2.0, 2.0]
            # translate = [0.5,-1.2,-0.2]
        # elif animation=='chair':
            # scale = [1.5, 1.5, 1.5]
            # translate = [1,-0.3,-0.3]+res_dist
        print_distance_result('Box', 'Box', res)

        animation='horse_holding'
        if animation=='horse':
            scale = [2.0, 2.0, 2.0]
            translate = [0.5,-1.2,-0.2]
        elif animation=='chair':
            scale = [1.5, 1.5, 1.5]
            # translate = [0.5,0.1,0.1]
            translate = [1,-0.3,-0.3]
        elif animation=='horse_holding':
            scale = [2.8, 2.8, 2.8]
            translate = [0.3,-1.7,0]

        st=trimesh.transformations.scale_and_translate(scale, translate)
        mesh_horse.apply_transform(st)
        # req = fcl.ContinuousCollisionRequest()
        # res = fcl.ContinuousCollisionResult()

        # dist = fcl.continuousCollide(fcl.CollisionObject(mesh_bvh, fcl.Transform()),
        #                                                           fcl.Transform(np.array([0.0, 0.0, 0.0])),
        #                                                           fcl.CollisionObject(mesh_horse_bvh, fcl.Transform()),
        #                                                           fcl.Transform(np.array([0.0, 0.0, 0.0])),
        #                                                           req, res)
        # print_continuous_collision_result('Box', 'Cylinder', res)
#
        # if mesh_filename is not None:
        print (mesh_filename)
        mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh_horse.apply_transform(R)

        # if mesh_filename is not None:
            # mesh_horse.export(mesh_filename)

        # if angle and axis:
            # R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            # mesh_horse.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000.
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        # print("-- Distance result: ", dis)
        # print(result.nearest_points)

        # while (not manager.in_collision_internal()):
            # manager, objects = trimesh.collision.scene_to_collision(mesh_scene)
            # print ("============>", manager,objects,manager.in_collision_internal())
            # scale = [1.5, 1.5, 1.5]
            # translate = [-0.5+0.1*10,-0.5+0.1*10,-0.5+0.1*10]
            # for i in range(20,1):
                # for j in range(20,1):
                    # for k in range(20,1):
                        # translate = [-0.5+0.1*i,-0.5+0.1*j,-0.5+0.1*k]
                        # st=trimesh.transformations.scale_and_translate(scale, translate)
                        # mesh_horse.apply_transform(st)
                        # manager, objects = trimesh.collision.scene_to_collision(mesh_scene)
                        # if ( manager.in_collision_internal()):
                            # break
                    # else:
                        # continue
                    # break
                # else:
                    # continue
                # break
                        # print ("============>", manager,objects,manager.in_collision_internal())

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        mesh_horse = pyrender.Mesh.from_trimesh(mesh_horse, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')
        mesh_horse_node = self.scene.add(mesh_horse, 'mesh_horse')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :,:-1] * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)



        self.scene.remove_node(mesh_node)
        self.scene.remove_node(mesh_horse_node)
        self.scene.remove_node(cam_node)
        # print (image)

        return image
    def render_horse(self, img, verts, cam, angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9]):
        # print (verts)

        # mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)
        mesh = trimesh.load("/data/swfcode/VIBE/horse/horse-gallop/horse-gallop-01.obj")

        # Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        Rx = trimesh.transformations.rotation_matrix(math.radians(50), [0, 1, 0.2])
        mesh.apply_transform(Rx)
        scale = [1.7, 1.7, 1.7]
        # translate = [0.5,-0.8,-1]
        translate = [-1,1,1]
        st=trimesh.transformations.scale_and_translate(scale, translate)
        mesh.apply_transform(st)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000.
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return image
