#LIBRERIAS
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.window import Window, key
from pyglet.gl import *
from pyglet.app import run
from pyglet import math
from pyglet import clock

import sys, os
import numpy as np
root = (os.path.dirname(__file__))

#MODULOS (cuidado con las rutas)
sys.path.append(os.path.dirname(os.path.dirname((os.path.dirname(__file__)))))
from librerias.helpers import init_axis, mesh_from_file
from librerias.camera import FreeCamera
from librerias.scene_graph import SceneGraph
from librerias import shapes
from librerias.drawables import Texture, Model, DirectionalLight, Material, PointLight

#Controla la ventana
class Controller(Window):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time = 0
        self.light_mode = False

#CAMARA definida en una clase
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

    controller = Controller(1000, 1000,"Auxiliar 8")
    controller.set_exclusive_mouse(True)

    # Pipelines 
    pipeline = ShaderProgram(Shader(vert_source, "vertex"), Shader(frag_source, "fragment"))
    color_pipeline = ShaderProgram(Shader(color_vert, "vertex"), Shader(color_frag, "fragment"))

    with open(root +  "/shaders/color_mesh_lit.vert") as f:
        color_vertex_source_code = f.read()

    with open(root +  "/shaders/color_mesh_lit.frag") as f:
        color_fragment_source_code = f.read()

    #Se define el pipeline
    l_vert_program = Shader(color_vertex_source_code, "vertex")
    l_frag_program = Shader(color_fragment_source_code, "fragment")
    light_pipeline = ShaderProgram(l_vert_program, l_frag_program)

  
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


    cam = MyCam([0,0,1])

    axis = init_axis(cam)

    world = SceneGraph(cam)
    world.add_node("scene")

  
    # Planicie
    tile_size = 2
    grid_size = 10  # 10x10
    for i in range(-grid_size // 2, grid_size // 2):
        for j in range(-grid_size // 2, grid_size // 2):
            tile_name = f"grass_{i}_{j}"
            world.add_node(tile_name, mesh=cube, texture=grass, pipeline=pipeline)
            world[tile_name]["position"] = [i * tile_size, -0.01, j * tile_size]
            world[tile_name]["scale"] = [tile_size, 0.01, tile_size]

    # Sphere
    world.add_node("sphere", attach_to="scene",
                   mesh=sphere, color=[*shapes.MAGENTA, 1.0],
                   pipeline=color_pipeline,
                   position=[0, 0.3, 0],
                   scale=[0.4, 0.4, 0.4])

    # Rock   
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
    

    # Sun
    world.add_node("sun", attach_to="scene",
               mesh=sphere,  
               color=[1.0, 1.0, 0.0, 1.0],  # Amarillo brillante
               pipeline=color_pipeline,
               position=[1.0, 6.5, 8.0],
               )
    
    # Sunlight
    world.add_node("sunlight",
                   attach_to="sun",
                   pipeline=light_pipeline,
                   light = PointLight(diffuse = [0.9, 0.1, 1], specular = [0.5, 0.5, 1], ambient = [0.1, 0.1, 0.5])
                   )

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
        

    #CAMARA vista en aux5
    @controller.event
    def on_key_press(symbol, modifiers):
        if symbol == key.W:
            cam.direction[0] = 1
        if symbol == key.S:
            cam.direction[0] = -1

        if symbol == key.A:
            cam.direction[1] = 1
        if symbol == key.D:
            cam.direction[1] = -1

    @controller.event
    def on_key_release(symbol, modifiers):
        if symbol == key.W or symbol == key.S:
            cam.direction[0] = 0

        if symbol == key.A or symbol == key.D:
            cam.direction[1] = 0

    @controller.event
    def on_mouse_motion(x, y, dx, dy):
        cam.yaw += dx * .001
        cam.pitch += dy * .001
        cam.pitch = math.clamp(cam.pitch, -(np.pi/2 - 0.01), np.pi/2 - 0.01)

    #Informacion que se actualiza con el tiempo
    def update(dt):
        world.update()
        axis.update()
        cam.time_update(dt)

        c_pos = cam.position.copy()
        c_pos[1] = 0

        controller.time += dt

    clock.schedule_interval(update,1/60)
    run()