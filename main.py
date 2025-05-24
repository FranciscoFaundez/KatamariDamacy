#LIBRERIAS
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.window import Window, key
from pyglet.gl import *
from pyglet.app import run
from pyglet import math
from pyglet import clock

import sys, os
import numpy as np
import random

root = (os.path.dirname(__file__))

#MODULOS (cuidado con las rutas)
sys.path.append(os.path.dirname(os.path.dirname((os.path.dirname(__file__)))))
from librerias.helpers import init_axis, mesh_from_file
from librerias.camera import FreeCamera
from librerias.scene_graph import SceneGraph
from librerias import shapes
from librerias.drawables import Texture, Model, DirectionalLight, Material, PointLight
import librerias.transformations as tr
from librerias.camera import OrbitCamera, FreeCamera, Camera


class Controller(Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = 0
        self.light_mode = False

class MyCam(FreeCamera):
    def __init__(self, position=np.array([0, 0, 0]), camera_type="perspective"):
        super().__init__(position, camera_type)
        self.direction = np.array([0,0,0])
        self.speed = 2

    def time_update(self, dt):
        self.update()
        dir = self.direction[0]*self.forward + self.direction[1]*self.right
        dir_norm = np.linalg.norm(dir)
        if dir_norm:
            dir /= dir_norm
        self.position += dir*self.speed*dt
        self.focus = self.position + self.forward

class FirstPersonCamera(FreeCamera):
    def __init__(self, target_node, camera_type="perspective"):
        self.target_node = target_node
        super().__init__([0, 0, 0], camera_type)

    def update(self):
        super().update()  # actualiza pitch, yaw, forward, right, up

        target_pos = self.target_node["position"]

        # Mantiene altura y adelanto, pero ahora respeta orientación 3D completa
        pos_offset = np.array([0.0, 0.17, 0.0]) + 0.15 * self.forward
        self.position = np.array(target_pos) + pos_offset
        self.focus = self.position + self.forward



class ThirdPersonCamera(FreeCamera):
    def __init__(self, target_node, offset=[0, 1.2, 3.5], camera_type="perspective"):
        self.target_node = target_node
        self.offset = np.array(offset, dtype=np.float32)
        super().__init__([0, 0, 0], camera_type)

    def update(self):
        super().update()
        target_pos = self.target_node["position"]
        backward = -self.forward * self.offset[2]
        up = np.array([0, self.offset[1], 0])
        self.position = np.array(target_pos) + backward + up
        self.focus = np.array(target_pos)

if __name__ == "__main__":

    controller = Controller(1000, 1000,"Auxiliar 8")
    controller.set_exclusive_mouse(True)


    with open(root +  "/shaders/color_mesh_lit.vert") as f:
        color_vertex_source_code = f.read()

    with open(root +  "/shaders/color_mesh_lit.frag") as f:
        color_fragment_source_code = f.read()

            
    with open(root +  "/shaders/textured_mesh_lit.vert") as f:
        text_vertex_source_code = f.read()

    with open(root +  "/shaders/textured_mesh_lit.frag") as f:
        text_fragment_source_code = f.read()

    #Se define el pipeline
    cl_vert_program = Shader(color_vertex_source_code, "vertex")
    cl_frag_program = Shader(color_fragment_source_code, "fragment")
    cl_pipeline = ShaderProgram(cl_vert_program, cl_frag_program)

    tl_vert_program = Shader(text_vertex_source_code, "vertex")
    tl_frag_program = Shader(text_fragment_source_code, "fragment")
    tl_pipeline = ShaderProgram(tl_vert_program, tl_frag_program)

  
    # --------------------------
    # Cargar modelos
    # --------------------------
    #Planicie
    grass = Texture(root + "/assets/grass.png", minFilterMode=GL_NEAREST, maxFilterMode=GL_NEAREST)
    face_uv = [0, 0, 1, 0, 1, 1, 0, 1]
    texcoords = face_uv * 6
    face_normals = [0, 1, 0] * 6 * 4 
    cube = Model(shapes.Cube["position"], texcoords, normal_data=face_normals, index_data=shapes.Cube["indices"])

    # Esfera
    sphere = mesh_from_file(root + "/assets/sphere.obj")[0]['mesh']

    # Objeto 1 (Roca)
    rock = mesh_from_file(root + "/assets/Rock_1_I_Color1.obj")

    # Objeto 2 (Árbol)
    tree = mesh_from_file(root + "/assets/Tree_1_A_Color1.obj")

    # Edificio
    building = mesh_from_file(root + "/assets/building/TallBuilding01.obj")
    building_text = Texture(root + "/assets/building/TallBuilding01.png")


    cam = MyCam([0,0,1])

    axis = init_axis(cam)

    world = SceneGraph(cam)

    

    world.add_node("scene")

  
    # Planicie
    tile_size = 1.5
    grid_size = 10  # 10x10
    for i in range(-grid_size // 2, grid_size // 2):
        for j in range(-grid_size // 2, grid_size // 2):
            tile_name = f"grass_{i}_{j}"
            world.add_node(tile_name, mesh=cube, texture=grass, pipeline=tl_pipeline)
            world[tile_name]["position"] = [i * tile_size, -0.01, j * tile_size]
            world[tile_name]["scale"] = [tile_size, 0.01, tile_size]
            world[tile_name]["material"] = Material(ambient=[0.3, 0.3, 0.3], diffuse=[0.3, 0.3, 0.3], specular=[0.3, 0.3, 0.3], shininess=10)

    # Sphere
    world.add_node("sphere", attach_to="scene",
                   mesh=sphere, color=[*shapes.MAGENTA, 1.0],
                   pipeline=cl_pipeline,
                   position=[0, 0.22, 0],
                   scale=[0.4, 0.4, 0.4],
                   material = Material(ambient=[0.3, 0.3, 0.3], diffuse=[0.3, 0.3, 0.3], specular=[0.3, 0.3, 0.3], shininess=10))

    models = [
        {"name": "rock", "mesh": rock[0]["mesh"], "texture": rock[0]["texture"], "scale": [0.1, 0.1, 0.1]},
        {"name": "tree", "mesh": tree[0]["mesh"], "texture": tree[0]["texture"], "scale": [0.5, 0.5, 0.5]},
        {"name": "building", "mesh": building[0]["mesh"], "texture": building_text, "scale": [0.5, 0.5, 0.5]},
    ]

    # Elije entre 6 y 10 elemento spara poner en pantalla
    num_objects = random.randint(6, 10)

    # Guardamos aquí los objetos que no han sido tocados y los que ya lo fueron
    absorvibles = []  
    absorbidos = set()

    # Ubica los distintos objetos en el piso
    for i in range(num_objects):
        obj = random.choice(models)
        pos_x = random.uniform(-grid_size // 2 + 1, grid_size // 2 - 1)
        pos_z = random.uniform(-grid_size // 2 + 1, grid_size // 2 - 1)
        y_offset = 0.05 if obj["name"] == "rock" else 0.4  # ajusta altura según objeto

        name = f"{obj['name']}_rand_{i}"
        absorvibles.append(name)
        world.add_node(name,
                    attach_to="scene",
                    mesh=obj["mesh"],
                    pipeline=tl_pipeline,
                    texture=obj["texture"],
                    position=[pos_x * tile_size, y_offset, pos_z * tile_size],
                    scale=obj["scale"],
                    material=Material())

    # Sunlight
    world.add_node("sunlight_color",
                   attach_to="scene",
                   pipeline=cl_pipeline,
                   light = DirectionalLight(diffuse = [0.9, 0.9, 0.1], specular = [0.9, 0.9, 0.1], ambient = [0.9, 0.9, 0.1]),
                   rotation=[-45, 45, 0]
                   )

    world.add_node("sunlight_tecture",
                   attach_to="scene",
                   pipeline=tl_pipeline,
                   light = DirectionalLight(diffuse = [0.9, 0.9, 0.1], specular = [0.9, 0.9, 0.1], ambient = [0.9, 0.9, 0.1]),
                   rotation=[-45, 45, 0]
                   )

    cam_orbit = OrbitCamera(6.775)
    cam_orbit.theta = 0.0001  # vista perfectamente desde arriba
    cam_orbit.update()


    cam_fp = FirstPersonCamera(world["sphere"])
    cam_tp = ThirdPersonCamera(world["sphere"], offset=[0, 0.8, 1.4])

    active_camera = cam_orbit  
    world.camera = active_camera

    @controller.event
    def on_draw():
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)
        controller.clear()
        glClearColor(1,1,1,1)
        #axis.draw()
        world.draw()


    # Variables globales para control
    pressed_keys = set()
    sphere_direction = np.array([0.0, 0.0])
    rotation_accum = 0  # acumulador de ángulo de rotación
     
    # Distancias desde el centro de la esfera a la que se deben pegar los objetos
    distance_by_type = {
    "tree": 0.8,
    "building": 0.8,
    "rock": 0.65
    }



    @controller.event
    def on_key_press(symbol, modifiers):
        global active_camera, sphere_direction
        pressed_keys.add(symbol)
        if symbol == key._1:
            active_camera = cam_orbit
        elif symbol == key._2:
            active_camera = cam_fp
        elif symbol == key._3:
            active_camera = cam_tp

        if symbol == key.W:
            sphere_direction[1] = 1
        elif symbol == key.S:
            sphere_direction[1] = -1
        elif symbol == key.A:
            sphere_direction[0] = 1
        elif symbol == key.D:
            sphere_direction[0] = -1

    @controller.event
    def on_key_release(symbol, modifiers):
        global sphere_direction
        pressed_keys.discard(symbol)
        if symbol in (key.W, key.S):
            sphere_direction[1] = 0
        if symbol in (key.A, key.D):
            sphere_direction[0] = 0

    @controller.event
    def on_mouse_motion(x, y, dx, dy):
        if isinstance(active_camera, FreeCamera):
            active_camera.yaw += dx * .001
            active_camera.pitch += dy * .001
            active_camera.pitch = np.clip(active_camera.pitch, -(np.pi/2 - 0.01), np.pi/2 - 0.01)


    def update(dt):
        global axis, sphere_direction
        world.camera = active_camera

        if isinstance(active_camera, FreeCamera):
            active_camera.update()

        speed = 2
        input_dir = np.array([sphere_direction[0], 0, sphere_direction[1]])
        if np.linalg.norm(input_dir) > 0:
            input_dir = input_dir / np.linalg.norm(input_dir)

        # Usar orientación de cámara
        if isinstance(active_camera, FreeCamera):
            forward = np.array(active_camera.forward)
            right = np.array(active_camera.right)
        else:
            # Calcular forward y right para OrbitCamera
            theta = active_camera.theta
            phi = active_camera.phi
            forward = -np.array([
                np.sin(theta) * np.sin(phi),
                0,
                np.sin(theta) * np.cos(phi)
            ])
            right = np.cross([0, 1, 0], forward)

        # Proyectar en plano XZ
        forward[1] = 0
        right[1] = 0
        forward = forward / np.linalg.norm(forward)
        right = right / np.linalg.norm(right)

        move_dir = input_dir[2] * forward + input_dir[0] * right
        if np.linalg.norm(move_dir) > 0:
            move_dir = move_dir / np.linalg.norm(move_dir)

            # Traslación
            pos = np.array(world["sphere"]["position"])
            delta = move_dir * speed * dt
            pos[0] += delta[0]
            pos[2] += delta[2]
            world["sphere"]["position"] = pos.tolist()

            # Rotación tipo rueda
            distance = np.linalg.norm(delta)
            if distance > 0:
                radius = 0.4 * 0.5  # aprox: escala x radio de la esfera
                angle_delta = 0.5 * distance / radius
                global rotation_accum
                rotation_accum += angle_delta

                # calcular eje de rotación (horizontal) usando producto cruzado
                rotation_axis = np.cross(move_dir, [0, 1, 0])
                if np.linalg.norm(rotation_axis) > 0:
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

                    # Para convertir un eje arbitrario en rotación XYZ (aprox)
                    # usamos solo rotación en X por simplicidad (como rueda)
                    world["sphere"]["rotation"] = [rotation_accum, 0, 0]

        # Detección de colisiones y absorción
        esfera_pos = np.array(world["sphere"]["position"])
        radio_esfera = 0.4 * 0.5  # mismo valor que usamos antes

        for name in absorvibles:
            if name in absorbidos:
                continue

            obj_pos = np.array(world[name]["position"])
            dist = np.linalg.norm(obj_pos - esfera_pos)

            if dist < radio_esfera + 0.3:  # umbral de colisión (ajustable)
                # Marcar como absorbido
                absorbidos.add(name)

                # Calcular vector desde centro esfera a objeto
                direccion_local = obj_pos - esfera_pos
                norma = np.linalg.norm(direccion_local)
                if norma < 1e-5:
                    direccion_local = np.array([1.0, 0.0, 0.0])  # dirección por defecto
                else:
                    direccion_local = direccion_local / norma

                # detectar tipo por nombre
                tipo = "rock"  # por defecto
                for clave in distance_by_type:
                    if clave in name:
                        tipo = clave
                        break

                r_contact = distance_by_type[tipo]
                world[name]["position"] = (direccion_local * r_contact).tolist()

                # pegar a la esfera
                world.graph.add_edge("sphere", name)


        axis.update()
        cam.time_update(dt)
        world.update()
        controller.time += dt


    clock.schedule_interval(update,1/60)
    run()