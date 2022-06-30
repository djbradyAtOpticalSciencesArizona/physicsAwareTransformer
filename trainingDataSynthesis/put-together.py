import bpy
import random
from math import radians
import math
import sys
sys.path.append("/home/qian/Documents/TrainingDataSynthesis")
from utils import *

mesh_obj_list = ["plane", "cube", "uv_sphere", "cylinder", "cone", "torus"]
tex_list = ["ShaderNodeTexVoronoi",
                    "ShaderNodeTexMagic",
                    "ShaderNodeTexBrick",
                    "ShaderNodeTexChecker"]
surface_list = ["ShaderNodeTexMusgrave", "ShaderNodeTexNoise"]

def gen_random_obj_with_texture(mesh_type = None):

    if mesh_type is None:
        mesh_type = random.choice(mesh_obj_list)
    obj = add_mesh_obj(mesh_type)
    if "Plane" in obj.name or "Cube" in obj.name:
        add_modifier(obj, 4)
    mat = add_material(obj)

    tex_type = random.choice(tex_list)
    add_texture(obj, mat, tex_type, tex_random_params(tex_type))
    
    surface_type = random.choice(surface_list)
    displace_surface(obj, mat, surface_type, surface_random_params(surface_type))
    return obj

def set_animation(obj, trans_params, is_background = False):
    if is_background is not True:
        obj.location, obj.rotation_euler, obj.scale = trans_params
    else:
        obj.location, _, _ = trans_params
    obj.keyframe_insert(data_path="location", index=-1)
    if is_background is not True:
        obj.keyframe_insert(data_path="rotation_euler", index=-1)
        obj.keyframe_insert(data_path="scale", index=-1)
    
def get_rand_trans():
    location = (random.random() * 4 - 2, 
                    random.random() * 4 - 2, 
                    random.random() * 4 - 2)
    rotation = (2 * random.random() * math.pi, 
                2 * random.random() * math.pi, 
                2 * random.random() * math.pi)
    scale =    (0.1 + 1.9 * random.random(), 
                0.1 + 1.9 * random.random(), 
                0.1 + 1.9 * random.random())
    return (location, rotation, scale)
                
def gen_random_animation(obj_list, patition_len = 7):
    '''
    obj_list: contains a list of mesh objects
    ---
    gen random partition inside n_frames, 
    e.g. n_frames = 60, partition = [0, 10, 21, 33, 45, 52, 60]
    '''
    scene = bpy.context.scene
    n_frames = scene.frame_end
    interval = int(n_frames/(patition_len - 1))
    for obj in obj_list:
        if obj.name != "background":
            partition = [interval*i for i in range(patition_len)] ## equally distributed
            for curr_time in partition:
                scene.frame_set(curr_time)
                trans_params = get_rand_trans()
                set_animation(obj, trans_params)
        else:
            partition = [interval*i for i in range(patition_len)]
            for curr_time in partition:
                scene.frame_set(curr_time)
                trans_params = get_rand_trans()
                set_animation(obj, trans_params, True)
    
if __name__ == '__main__':
    random.seed("qian038")
    ''' initialize scene '''
#    init_scene(res = (4096 * 4, 4096 * 4), n_frames = 240, use_gpu = True,
#                n_samples = 1024, render_region = True, 
#                render_params = (4096 * 2 - 128, 4096 * 2 + 128, 
#                               4096 * 2 - 128, 4096 * 2 + 128))
    init_scene_eevee((2048, 1536), 60)
    
    n_scenes = 900
    
    for scene_idx in range(n_scenes):
        ''' clear scene '''
        clear_scene()

        ''' add light source '''
        add_light((-3, 0, 7), light_type = 'AREA')

        ''' set camera '''
        add_camera(loc = (7.35889, -6.92579, 4.95831), lens=64) #, 256
        
        ''' add background cube '''
        background = gen_random_obj_with_texture("cube")
        background.dimensions = (20, 20, 20)
        background.name = "background"
        
        ''' add objs '''
        n_obj = random.randint(10, 20)
        obj_list = [background]
        for i in range(n_obj):
            obj_list.append(gen_random_obj_with_texture())

        gen_random_animation(obj_list, 26)
        bpy.context.scene.frame_current = 5

#        ''' output '''
        path = '/home/qian/Downloads/blender_2k_test/scene{:04d}/'.format(scene_idx)
        link_file_node(path + 'Image', 'Image', 'PNG', '8')
#        link_file_node(path + 'Depth', 'Depth')
#        link_file_node(path + 'Vector', 'Vector')
        bpy.context.scene.frame_end = 5#30
        bpy.ops.render.render(animation = False)
##        bpy.context.scene.frame_end = 240
        clear_output_nodes()
    