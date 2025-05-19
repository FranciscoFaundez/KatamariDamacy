import pyglet
from pyglet.gl import *
from pyglet.math import Mat4, Vec3
from pyglet.window import key
from pyglet.graphics.shader import Shader, ShaderProgram
import numpy as np
import os
# Agregar mas si es necesario

# Rutas
root = (os.path.dirname(__file__))

# Librerías propias
from librerias.scene_graph import *
from librerias.helpers import mesh_from_file
from librerias.drawables import Model
from librerias import shapes

# --------------------------
# Ventana principal
# --------------------------
class Controller(pyglet.window.Window):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time = 0.0
        self.gameState = 0
        self.sphere_speed = 2.0  # velocidad en unidades por segundo
        self.sphere_rotation = [0.0, 0.0, 0.0]  # rotaciones acumuladas
        self.pressed_keys = set()
        self.camera_mode = 1  # 1: aérea fija, 2: primera persona, 3: tercera persona
        self.yaw = 0.0
        self.pitch = 0.0
        self.mouse_sensitivity = 0.25
        self.set_exclusive_mouse(True)




WIDTH = 1000
HEIGHT = 1000
window = Controller(WIDTH, HEIGHT, "Template T3")

# --------------------------
# Shaders
# --------------------------
if __name__ == "__main__":

# Shader con textura
    vert_source = """
#version 330

in vec3 position;
in vec2 texCoord; 

out vec2 fragTexCoord; 

uniform mat4 u_model = mat4(1.0);
uniform mat4 u_view = mat4(1.0);
uniform mat4 u_projection = mat4(1.0);

void main() {
    fragTexCoord = texCoord;
    gl_Position = u_projection * u_view * u_model * vec4(position, 1.0f);
}
    """
    frag_source = """
#version 330
in vec2 fragTexCoord;

uniform sampler2D u_texture;

out vec4 outColor;

void main() {
    outColor = texture(u_texture, fragTexCoord);
}
    """

# Shader con color sólido
    color_vert = """
#version 330
in vec3 position;

uniform mat4 u_model = mat4(1.0);
uniform mat4 u_view = mat4(1.0);
uniform mat4 u_projection = mat4(1.0);

void main() {
    gl_Position = u_projection * u_view * u_model * vec4(position, 1.0f);
}
"""

    color_frag = """
#version 330
uniform vec4 u_color;
out vec4 outColor;

void main() {
    outColor = u_color;
}
"""

    # Pipelines 
    pipeline = ShaderProgram(Shader(vert_source, "vertex"), Shader(frag_source, "fragment"))
    color_pipeline = ShaderProgram(Shader(color_vert, "vertex"), Shader(color_frag, "fragment"))

    # --------------------------
    # Cargar modelos
    # --------------------------
    #Planicie
    grass = Texture(root + "/assets/grass.png", minFilterMode=GL_NEAREST, maxFilterMode=GL_NEAREST)
    face_uv = [0, 0, 1, 0, 1, 1, 0, 1]
    texcoords = face_uv * 6
    cube = Model(shapes.Cube["position"], texcoords, index_data=shapes.Cube["indices"])

    # Esfera
    sphere = mesh_from_file(root + "/assets/sphere.obj")[0]['mesh']

    # Objeto 1 (Roca)
    rock = mesh_from_file(root + "/assets/Rock_1_I_Color1.obj")

    # Objeto 2 (Árbol)
    tree = mesh_from_file(root + "/assets/Tree_1_A_Color1.obj")

    # Edificio
    building = mesh_from_file(root + "/assets/building/TallBuilding01.obj")
    building_text = Texture(root + "/assets/building/TallBuilding01.png")

    # --------------------------
    # Grafo de escena
    # --------------------------
    world = SceneGraph()
    world.add_node("scene")

    #Planicie
    tile_size = 1.0
    grid_size = 10  # 10x10
    for i in range(-grid_size // 2, grid_size // 2):
        for j in range(-grid_size // 2, grid_size // 2):
            tile_name = f"grass_{i}_{j}"
            world.add_node(tile_name, mesh=cube, texture=grass, pipeline=pipeline)
            world[tile_name]["position"] = [i * tile_size, -0.01, j * tile_size]
            world[tile_name]["scale"] = [tile_size, 0.01, tile_size]

    #Esfera
    world.add_node("sphere", attach_to="scene",
                   mesh=sphere, color=[*shapes.MAGENTA, 1.0],
                   pipeline=color_pipeline,
                   position=[0, 0.3, 0],
                   scale=[0.4, 0.4, 0.4])

    #Rock   
    world.add_node("rock", attach_to="scene")
    world.add_node("rock_model", 
                   attach_to = "rock", 
                   mesh = rock[0]["mesh"], 
                   pipeline = pipeline, 
                   texture = rock[0]["texture"],
                   position = [0.5, 0.05, 0.5],
                   scale = [0.1, 0.1, 0.1])

    # Tree
    world.add_node("tree", attach_to="scene")
    world.add_node("tree_model", 
                   attach_to= "tree",
                   mesh = tree[0]["mesh"],
                   pipeline=pipeline, 
                   texture = tree[0]["texture"],
                   position = [0.2, 0.4, 0.6],
                   scale=[0.5, 0.5, 0.5])
    

    # Building
    world.add_node("building", attach_to="scene")
    world.add_node("building_model", 
                   attach_to= "building",
                   mesh = building[0]["mesh"],
                   pipeline=pipeline, 
                   texture = building_text,
                   position = [0.8, 0.45, 0.8],
                   scale=[0.5, 0.5, 0.5])

    # --------------------------
    # Cámara
    # --------------------------
    # Cámara aérea fija
    eye = Vec3(0.0, 4.0, 0.0)       # Cámara desde arriba
    target = Vec3(0.0, 0.0, 0.0)    # Centro de la escena
    up = Vec3(0.0, 0.0, -1.0)       # "Arriba" apunta hacia -Z para mantener orientación horizontal

    pipeline["u_view"] = Mat4.look_at(eye, target, up)
    pipeline["u_projection"] = Mat4.perspective_projection(WIDTH / HEIGHT, 0.01, 100, 90)

    color_pipeline["u_view"] = Mat4.look_at(eye, target, up)
    color_pipeline["u_projection"] = Mat4.perspective_projection(WIDTH / HEIGHT, 0.01, 100, 90)

    # --------------------------
    # Update y render
    # --------------------------
    #pressed_keys = window.pressed_keys


    @window.event
    def on_mouse_motion(x, y, dx, dy):
        if window.camera_mode != 1:  # Solo afecta si no es cámara aérea
            window.yaw -= dx * window.mouse_sensitivity
            window.pitch += dy * window.mouse_sensitivity
            window.pitch = max(-89.0, min(89.0, window.pitch))

    @window.event
    def on_key_press(symbol, modifiers):
        window.pressed_keys.add(symbol)
        if symbol == key._1:
            window.camera_mode = 1
        elif symbol == key._2:
            window.camera_mode = 2
        elif symbol == key._3:
            window.camera_mode = 3


    @window.event
    def on_key_release(symbol, modifiers):
            window.pressed_keys.discard(symbol)

    def update(dt):
        world.update()

        if window.camera_mode == 1:
            # Cámara aérea fija
            forward = Vec3(0, 0, -1)
            right = Vec3(1, 0, 0)
        else:
            # Cámara 2 o 3: movimiento relativo a la rotación de la cámara (yaw)
            forward = Vec3(-np.sin(np.radians(window.yaw)), 0, -np.cos(np.radians(window.yaw)))
            right = Vec3(-forward.z, 0, forward.x)

        direction = Vec3(0, 0, 0)
        if key.W in window.pressed_keys:
            direction += forward
        if key.S in window.pressed_keys:
            direction -= forward
        if key.A in window.pressed_keys:
            direction -= right
        if key.D in window.pressed_keys:
            direction += right

        # Si hay dirección, mover la esfera
        if direction.length() > 0:
            direction = direction.normalize()

            # Movimiento
            velocity = direction * window.sphere_speed * dt
            pos = world["sphere"]["position"]
            world["sphere"]["position"] = [
                pos[0] + velocity.x,
                pos[1],
                pos[2] + velocity.z
            ]

            # Rotación simulada (como si rodara)
            radius = 0.4
            distance = velocity.length()
            angle_deg = (distance / radius) * (180 / np.pi)
            rotation_axis = Vec3(direction.z, 0, -direction.x)

            if abs(rotation_axis.x) > 0:
                window.sphere_rotation[2] += angle_deg * np.sign(rotation_axis.x)
            if abs(rotation_axis.z) > 0:
                window.sphere_rotation[0] += angle_deg * np.sign(rotation_axis.z)

            world["sphere"]["rotation"] = window.sphere_rotation


        # Actualizar cámara según el modo
        if window.camera_mode == 1:
            # Cámara aérea fija
            eye = Vec3(0.0, 4.0, 0.0)
            target = Vec3(0.0, 0.0, 0.0)
            up = Vec3(0.0, 0.0, -1.0)

        elif window.camera_mode == 2:
            # Primera persona: cámara dentro de la esfera, mira en la dirección del mouse
            sphere_pos = Vec3(*world["sphere"]["position"])
            front = Vec3(
                -np.sin(np.radians(window.yaw)) * np.cos(np.radians(window.pitch)),
                np.sin(np.radians(window.pitch)),
                -np.cos(np.radians(window.yaw)) * np.cos(np.radians(window.pitch))
            ).normalize()
            eye = sphere_pos + Vec3(0, 0.2, 0)
            target = eye + front
            up = Vec3(0, 1, 0)

        elif window.camera_mode == 3:
            sphere_pos = Vec3(*world["sphere"]["position"])
            behind = Vec3(
                np.sin(np.radians(window.yaw)),
                -0.1,  
                np.cos(np.radians(window.yaw))
            ).normalize()
            camera_distance = 1.2 
            height_offset = 0.6    
            eye = sphere_pos + behind * camera_distance + Vec3(0, height_offset, 0)
            target = sphere_pos + Vec3(0, 0.3, 0)
            up = Vec3(0, 1, 0)

        # Actualizar matrices de vista en los shaders
        view_matrix = Mat4.look_at(eye, target, up)
        pipeline["u_view"] = view_matrix
        color_pipeline["u_view"] = view_matrix


    @window.event
    def on_draw():
        glEnable(GL_DEPTH_TEST) 
        glEnable(GL_CULL_FACE)
        glClearColor(0.63, 0.6, 0.8, 0.0) # Color fondo
        window.clear()
        world.draw()

    pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()