import bpy
from math import radians
import random

def init_scene(res = (3840, 2160), n_frames = 240, n_samples = 4096, use_gpu = False, 
                render_region=True, render_params = None, render_tile = 256):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU' if use_gpu else 'CPU'
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    if isinstance(res, int):
        scene.render.resolution_x = res
        scene.render.resolution_y = res
    else:
        scene.render.resolution_x = res[0]
        scene.render.resolution_y = res[1]
    scene.render.tile_x = render_tile if use_gpu else 8
    scene.render.tile_y = render_tile if use_gpu else 8
    scene.cycles.samples = n_samples
    scene.cycles.preview_samples = 1
    scene.cycles.max_bounces = 1
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transparent_max_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.sample_clamp_indirect = 0
    scene.cycles.filter_width = 1.5
    scene.frame_end = n_frames
    if render_region:
        scene.render.use_border = True
        scene.render.use_crop_to_border = True
        x1, x2, y1, y2 = render_params
        scene.render.border_min_x = x1/scene.render.resolution_x
        scene.render.border_max_x = x2/scene.render.resolution_x
        scene.render.border_min_y = y1/scene.render.resolution_y
        scene.render.border_max_y = y2/scene.render.resolution_y
    else:
        scene.render.use_border = False
        
def init_scene_eevee(res = 512, n_frames = 240, anti_aliasing=True, color_correction=False):
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 16
    scene.eevee.taa_samples = 1
    scene.eevee.use_taa_reprojection = False
    if isinstance(res, int):
        scene.render.resolution_x = res
        scene.render.resolution_y = res
    else:
        scene.render.resolution_x = res[0]
        scene.render.resolution_y = res[1]
    scene.frame_end = n_frames
    if not anti_aliasing:
        scene.render.filter_size = 0
    if not color_correction:
        scene.display_settings.display_device = 'None'
        scene.sequencer_colorspace_settings.name = 'Linear'
    
def clear_scene(clear_mesh_only = False):
    data = bpy.data
    for ob in data.objects:
        if ob.type == 'MESH' or ob.type == 'EMPTY':
            data.objects.remove(data.objects[ob.name], do_unlink = True)
    if clear_mesh_only is False:
        for ob in data.cameras:
            data.cameras.remove(data.cameras[ob.name], do_unlink = True)
        for ob in data.lights:
            data.lights.remove(data.lights[ob.name], do_unlink = True)

def wavelength_to_rgb(wavelength, gamma=1):

    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    '''
    if isinstance(wavelength, str):
        if wavelength == 'white':
            return 1.0, 1.0, 1.0
        else:
            raise NotImplementedError("color not implemented")

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return R, G, B
    
def add_light(loc, energy = 10000, light_type = 'POINT', light_color=(1.0, 1.0, 1.0)):
    '''
    loc: 3-element tuple
    '''
    lamp_data = bpy.data.lights.new(name="light", type=light_type)  
    lamp_object = bpy.data.objects.new(name="light_obj", object_data=lamp_data)  
    bpy.context.collection.objects.link(lamp_object)  
    #lamp_object.location = (-3, 0, 7)
    lamp_object.location = loc
    lamp = bpy.data.lights[lamp_data.name]
    lamp.energy = energy
    lamp.color = light_color
    lamp.use_shadow = True
    if light_type == 'AREA':
        lamp.size = 15
        lamp.cycles.cast_shadow = False
        lamp.cycles.use_multiple_importance_sampling = False
    return lamp

    
def add_camera(loc = (0, 0, 0), rot=(0, 0, 0), lens = 6400, name="camera_obj", obj = None):
    '''
    loc: 3-element tuple
    rot: 3-element tuple in radians
    '''
    if obj is None:
        bpy.ops.object.empty_add()
        obj = bpy.context.object
    cam_data = bpy.data.cameras.new(name="camera")  
    cam_ob = bpy.data.objects.new(name=name, object_data=cam_data)  
    bpy.context.collection.objects.link(cam_ob)   
    cam_ob.location = loc 
    cam_ob.rotation_euler = rot
    cam = bpy.data.cameras[cam_data.name]  
    cam.lens = lens
    ### modified 04/30/21
    cam.sensor_width = 36
    bpy.context.scene.camera = cam_ob
    cam_ob.constraints.new("TRACK_TO")
    cam_ob.constraints["Track To"].target = obj
    return cam_ob

def add_array_cameras(locs = None, fs = None, obj = None):
    if obj is None:
        bpy.ops.object.empty_add()
        obj = bpy.context.object
    bpy.context.scene.render.use_multiview = True
    bpy.context.scene.render.views_format = 'MULTIVIEW'
    bpy.context.scene.render.views["left"].use = False
    bpy.context.scene.render.views["right"].use = False
    
    if locs is None:
        locs = [(10, 0, 0),
                (10, 0, -1), (10, 0, 1),
                (10, -1, 0), (10, 1, 0)]
        fs = [64] * len(locs)
    for i in range(len(locs)):
        cam = add_camera(locs[i], name = "camera_obj{}".format(i), lens = fs[i])
        cam.constraints.new("TRACK_TO")
        cam.constraints["Track To"].target = obj
        bpy.ops.scene.render_view_add()
        view_name = "RenderView" if i == 0 else "RenderView.{:03d}".format(i)
        bpy.context.scene.render.views[view_name].camera_suffix = "_obj{}".format(i)
    
    
def add_mesh_obj(type, params = {}, is_modifier = True):
    '''                                                               
    type: plane, cube, cylinder, cone, uv_sphere, torus             
    params: dict                                                           
    '''
    if type == "plane":
        bpy.ops.mesh.primitive_plane_add(**params)
    elif type == "cube":
        bpy.ops.mesh.primitive_cube_add(**params)
    elif type == "uv_sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(**params)
    elif type == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(**params)
    elif type == "cone":
        bpy.ops.mesh.primitive_cone_add(**params)
    elif type == "torus":
        bpy.ops.mesh.primitive_torus_add(**params)
    else:
        raise NotImplementedError
    obj = bpy.context.object
    return obj

def add_modifier(obj, level = 2):
    subsurf_mod = obj.modifiers.new(name = "Subdivision",type='SUBSUR\
F')
    subsurf_mod.subdivision_type = 'SIMPLE'
    subsurf_mod.show_only_control_edges = False
    subsurf_mod.levels = level
    subsurf_mod.render_levels = level

def add_material(obj):
    mat = bpy.data.materials.new("mat_" + str(obj.name))
    mat.use_nodes = True
    obj.data.materials.append(mat)
    return mat

def add_texture(obj, mat, type, params = None):
    '''                                                                       
    obj: type bpy_types.Object, can be assigned by bpy.context.object         
    type/params: texture type/params                                          
        supports modes:                                                       
        * ShaderNodeTexVoronoi, dict...                                       
        * ShaderNodeTexMagic,                                                 
        * ShaderNodeTexBrick,                                                 
        * ShaderNodeTexChecker                                                
        * ...                                                                 
    '''
    matnodes = mat.node_tree.nodes
    tex = matnodes.new(type)
    if type == "ShaderNodeTexVoronoi":
        tex.voronoi_dimensions = '4D'
    if type == "ShaderNodeTexMagic":
        tex.turbulence_depth = random.randint(1, 5)
    if params is not None:
        for name in params:
            tex.inputs[name].default_value = params[name]
    base_color = matnodes['Principled BSDF'].inputs['Base Color']
    mat.node_tree.links.new(base_color, tex.outputs['Color'])

    coord = matnodes.new("ShaderNodeTexCoord")
    mat.node_tree.links.new(coord.outputs['UV'], tex.inputs['Vector'])

def tex_random_params(type):
    params = {}
    if type == "ShaderNodeTexVoronoi":
        params['Scale'] = 10 + 20 * random.random()
        params['W'] = random.randint(-100, 100) * 0.2
    elif type == "ShaderNodeTexMagic":
        params['Scale'] = 10 + 20 * random.random()
        params['Distortion'] = 0.5 + 2.5 * random.random()
    elif type == "ShaderNodeTexBrick":
        params['Scale'] = 10 + 20 * random.random()
        params['Color1'] = (random.random(),random.random(),random.random(),1)
        params['Color2'] = (random.random(),random.random(),random.random(),1)
        params['Mortar'] = (random.random(),random.random(),random.random(),1)
        params['Mortar Size'] = random.random() * 0.1
    elif type == "ShaderNodeTexChecker":
        params['Scale'] = 40 + 20 * random.random()
        params['Color1'] = (random.random(),random.random(),random.random(),1)
        params['Color2'] = (random.random(),random.random(),random.random(),1)
    else:
        raise NotImplementedError
    return params

def displace_surface(obj, mat, type = "ShaderNodeTexMusgrave", 
                        params = None, disp_method = "BOTH"):                                                     
    matnodes = mat.node_tree.nodes
    dispnode = matnodes.new("ShaderNodeDisplacement")
    if disp_method == "BOTH":
        dispnode.inputs["Scale"].default_value = 0.3 * random.random()
    disp = matnodes['Material Output'].inputs['Displacement']
    mat.node_tree.links.new(disp, dispnode.outputs['Displacement'])

    ## ShaderNodeTexNoise["Fac"]                                              
    ## ShaderNodeTexMusgrave[0]                                               
    tex = matnodes.new(type)
    if type == "ShaderNodeTexMusgrave":
        tex.musgrave_dimensions = '4D'
    elif type == "ShaderNodeTexNoise":
        tex.noise_dimensions = '4D'
    else:
        raise NotImplementedError
    if params is not None:
        for name in params:
            tex.inputs[name].default_value = params[name]
    mat.node_tree.links.new(tex.outputs[0], dispnode.inputs['Height'])

    coord = matnodes.new("ShaderNodeTexCoord")
    mat.node_tree.links.new(coord.outputs['Object'], tex.inputs['Vector'])
    mat.cycles.displacement_method = disp_method
    
def surface_random_params(type = "ShaderNodeTexMusgrave"):
    params = {}
    if type == "ShaderNodeTexNoise":
        params['W'] = random.randint(-100, 100) * 0.2
        params['Scale'] = 10 + 20 * random.random()
        params['Detail'] = 1 + 2 * random.random()
        params['Distortion'] = -5 + 10 * random.random()
    elif type == "ShaderNodeTexMusgrave":
        params['W'] = random.randint(-100, 100) * 0.2
        params['Scale'] = 10 + 20 * random.random()
        params['Detail'] = 1 + 2 * random.random()
        params['Lacunarity'] = 5 * random.random()
    else:
        raise NotImplementedError
    return params

def link_file_node(base_path, output_type, 
                    format = 'OPEN_EXR', color_depth = '16', color_mode='RGB'):
    '''                                                                       
    scene: bpy.context.scene                                                  
    base_path: output path to store image files                               
    output_type: 'Vector'/'Depth'/'Image'                                     
    format: 'PNG'/'OPEN_EXR'                                                  
    color_depth: default '16' bits for png files and '16' bits half float for\
 OpenExr files                                                                
    '''
    scene = bpy.context.scene
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    render_layers = nodes['Render Layers']
    ## assure render layer node has property vector                           
    scene.view_layers["ViewLayer"].use_pass_vector = True

    file_node = nodes.new("CompositorNodeOutputFile")
    file_node.base_path = base_path
    file_node.format.file_format = format
    file_node.format.color_mode = color_mode
    file_node.format.color_depth = color_depth
    scene.node_tree.links.new(render_layers.outputs[output_type],
                            file_node.inputs['Image'])
                            
def clear_output_nodes(use_multiview=None):
    scene = bpy.context.scene
    nodes = scene.node_tree.nodes
    for node in nodes:
        if 'File Output' in node.name:
            nodes.remove(node)
    if use_multiview is None:
        use_multiview=bpy.context.scene.render.use_multiview
    if use_multiview:
        views = bpy.context.scene.render.views
        for view in views:
            if 'RenderView' in view.name:
                views.remove(view)

if __name__ == "__main__":
    random.seed("0038")
    init_scene_eevee(512, 2, anti_aliasing=False, color_correction=False)
    clear_scene()
    add_light((-3, 0, 7))
#    add_camera((7, -7, 5), (radians(63.6), 0, radians(46.7)), 64)
    add_array_cameras()
    obj = add_mesh_obj("uv_sphere", {})
    add_modifier(obj, 4)
    mat = add_material(obj)
    add_texture(obj, mat, "ShaderNodeTexBrick", 
                        tex_random_params("ShaderNodeTexChecker"))
    displace_surface(obj, mat, "ShaderNodeTexMusgrave",
                        surface_random_params("ShaderNodeTexMusgrave"))
    link_file_node('/home/qian/Desktop/test/Image/', 'Image', 'PNG', '8', 'BW')
    link_file_node('/home/qian/Desktop/test/Depth/', 'Depth')
    bpy.ops.render.render(animation = False)
    clear_output_nodes()
