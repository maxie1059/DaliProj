import drjit as dr
import mitsuba as mi
import numpy as np
import cv2

mi.set_variant('cuda_ad_spectral')

from mitsuba import ScalarTransform4f as T

class Hologram_Scene:
    def __init__(self):
        self.scene_dict = {'type': 'scene'}
        # Emitters
        self.emitter_locations = []
        self.emitter_max_radiance = 200.0
        # Cameras
        self.number_of_cam = 1
        self.cam_init_location = [15,0,0]
        self.cam_target = [0,0,0]
        self.cam_numerical_aperture = 0.01
        self.focal_length = 15.0
        self.angularFOV = 40.0
        self.img_resolution = [1024,1024]
        self.sample_per_pixels = 128
        self.img_type = 'rgb'
        self.total_viewing_angle = 120.0
        self.sensors = []
        # Glass
        self.glass_location = [0,0,0]
        self.glass_size = [4,6,0.1]
        self.glass_bsdf = 'dielectric'
        self.glass_ior = 'bk7'
        self.glass_roughness = 0.01
        self.glass_specular_reflectance = [0.6,0.6,0.6]
        self.max_viewing_angle = 90.0
        self.half_hologram_theta_max = self.max_viewing_angle/2
        self.amt_of_dust = 150
        # Objects
        self.obj_dir = []
        self.obj_type = []
        self.obj_bsdf = []
        self.obj_color = []
        self.obj_scale_factor = []
        self.obj_rotation = []
        self.obj_location = []
        # Threshold for object scaling
        self.obj_max_size_threshold = 12.0
        self.obj_min_size_threshold = 0.5
        self.obj_downscale_resolution = 2.0
        self.obj_upscale_resolution = 2.5

    
    def light_source_radiance(self,theta):
        return self.emitter_max_radiance * np.sinc(theta/self.half_hologram_theta_max)**2
    
    def create_emitters(self):
        for i,emitter_loc in enumerate(self.emitter_locations):
            emitter_name = 'emitter' + str(i+1)
            self.scene_dict[emitter_name] = {
                'type': 'point',
                'to_world': T().translate(emitter_loc),
                'intensity': {
                    'type': 'spectrum',
                    'value': 0.0
                }
            }
        
        emitter_left_glass = {
            'type': 'point',
            'to_world': T().translate([0.5, -3, 0]),
            'intensity': {
            'type': 'spectrum',
            'value': self.emitter_max_radiance
            }
        }
        emitter_right_glass = {
            'type': 'point',
            'to_world': T().translate([0.5, 3, 0]),
            'intensity': {
            'type': 'spectrum',
            'value': self.emitter_max_radiance
            }
        }
        self.scene_dict['emitter_left_glass'] = emitter_left_glass
        self.scene_dict['emitter_right_glass'] = emitter_right_glass

    def create_sensors(self):
        self.sensors.clear()
        for i in range(self.number_of_cam):
            if self.number_of_cam == 1:
                angle = 0.0
            else:
                start = -self.total_viewing_angle/2
                step = self.total_viewing_angle / (self.number_of_cam-1)
                angle = start + i*step
            sensor_rotation = T().rotate([0, 0, 1], angle)
            sensor_to_world = T().look_at(target=self.cam_target, origin=self.cam_init_location, up=[0, 0, 1])
            loading_sensor = mi.load_dict({
                'type': 'thinlens',
                "fov_axis": "x",
                "fov": self.angularFOV,
                "to_world": sensor_rotation @ sensor_to_world,
                'aperture_radius': self.cam_numerical_aperture,
                'focus_distance': self.focal_length,
                "film": {
                    "type": "hdrfilm",
                    "width": self.img_resolution[0],
                    "height": self.img_resolution[1],
                    "pixel_format": self.img_type,
            }})
            self.sensors.append((loading_sensor,angle))
    
    def find_max_min(self,value_list):
        max_val = value_list[0]
        min_val = value_list[0]
        for num in value_list:
            if num > max_val:
                max_val = num
            if num < min_val:
                min_val = num
        return max_val, min_val
    
    def create_obj(self):
        for i,dir in enumerate(self.obj_dir):
            normalized_scale = []
            downscale_value = 1.0
            upscale_value = 1.0
            obj_name = 'obj' + str(i+1)
            bsdf = {'type': self.obj_bsdf[i]}
            match self.obj_bsdf[i]:
                case 'diffuse':
                    bsdf['reflectance'] = {'type': 'rgb', 'value': self.obj_color[i]}
                case 'roughplastic':
                    bsdf['diffuse_reflectance'] = {'type': 'rgb', 'value': self.obj_color[i]}
            
            #Normalized object size
            scaleCheck = mi.load_dict({
                'type': self.obj_type[i],
                'filename': dir,
                'to_world': T().translate([0, 0, 0])}
            )
            bbox = scaleCheck.bbox()
            size = bbox.extents()  # Object’s dimensions
            max,min = self.find_max_min(size)
            if max > self.obj_max_size_threshold:
                #downscale
                downscale_value = self.obj_downscale_resolution/min
            if max < self.obj_min_size_threshold:
                #upscale
                upscale_value = self.obj_upscale_resolution/min
            for scale_factor in self.obj_scale_factor[i]:
                normalized_scale.append(scale_factor*downscale_value*upscale_value)
            
            self.scene_dict[obj_name] = {
                'type': self.obj_type[i],
                'filename': dir,
                'to_world': T().translate([-self.obj_location[i][0],self.obj_location[i][1],self.obj_location[i][2]])
                @ T().scale(normalized_scale)
                @ T().rotate(axis=(1, 0, 0), angle=self.obj_rotation[i][0])
                @ T().rotate(axis=(0, 1, 0), angle=self.obj_rotation[i][1])
                @ T().rotate(axis=(0, 0, 1), angle=self.obj_rotation[i][2]),
                'bsdf': bsdf
            }

    def create_glass(self):
        # Dust image pre-processing
        image = cv2.imread('Dust-Textures-03.jpg',0).astype(float)
        _,normalizedImg = cv2.threshold(image,self.amt_of_dust,255,cv2.THRESH_TOZERO)
        normalizedImg = cv2.normalize(normalizedImg, None, norm_type=cv2.NORM_MINMAX)
        normalizedImg = mi.Bitmap(normalizedImg)

        glass_material = {
            'type': self.glass_bsdf,
            'int_ior': self.glass_ior,
            'specular_reflectance': {
                'type': 'rgb',
                'value': self.glass_specular_reflectance
            },
        }
        if self.glass_bsdf == 'roughdielectric':
            glass_material['alpha'] = self.glass_roughness

        holographic_film = {
            'type': 'blendbsdf',
            'weight':{
                'type': 'bitmap',
                'bitmap': normalizedImg,
                'raw': True,
            },
            'bsdf_0': glass_material,
            'bsdf_1':{
                'type': 'bumpmap',
                'normalmap': {
                    'type': 'bitmap',
                    'raw': True,
                    'bitmap': normalizedImg
                },
            'bsdf': {
                'type': 'diffuse'
            },
            },
        }
        medium = {
            'type': 'homogeneous',
            'sigma_t': {
                'type': 'rgb',
                'value': [0.1, 0.1, 0.1] # Extinction coefficient
            },
            'albedo': {
                'type': 'rgb',
                'value': [0.9, 0.9, 0.9]  # Scattering albedo
            },
        }
        scale_factor = np.array(self.glass_size)
        glass_slab = {
            "type": "cube",
            "to_world": T().rotate(axis=(0, 1, 0), angle=90)
            @ T().translate([0, 0, 0])
            @ T().scale(scale_factor),
            'bsdf': {'type': 'ref', 'id': 'holographic-film'},
            'interior': {'type': 'ref','id':'medium'},
        }
        outerFrameColor = [0.0,0.0,0.0]
        outerFrameLeft = {
            "type": "cube",
            "to_world": T().rotate(axis=(0, 1, 0), angle=90)
            @ T().translate([0, scale_factor[1]*2, -scale_factor[2]*2])
            @ T().scale(scale_factor),
            'bsdf': {'type': 'diffuse', 'reflectance': {'type':'rgb','value':outerFrameColor}},
        }
        outerFrameRight = {
            "type": "cube",
            "to_world": T().rotate(axis=(0, 1, 0), angle=90)
            @ T().translate([0, -scale_factor[1]*2, -scale_factor[2]*2])
            @ T().scale(scale_factor),
            'bsdf': {'type': 'diffuse', 'reflectance': {'type':'rgb','value':outerFrameColor}},
        }
        self.scene_dict['holographic-film'] = holographic_film
        self.scene_dict['medium'] = medium
        self.scene_dict['slab'] = glass_slab
        self.scene_dict['outerFrameLeft'] = outerFrameLeft
        self.scene_dict['outerFrameRight'] = outerFrameRight

    def generate_hologram_imgs(
            self,
            objs_info,
            # Define each object a list of info
            # 1. A string of object directory. Ex: '/mitsuba-tutorials/scenes/meshes/bunny.ply'
            # 2. A string of object material: Ex: 'diffuse'
            # 3. A list of object colors using normalized RGB value [0,1]. Ex: [0.3, 0.4, 0.5]
            # 4. A list of integers of object scale factor in [x,y,z]. Ex: [2,30,10]
            # 5. A list of integer values for angles that the object will rotated in consecutive order x,y,z. Ex: [80,90,0] meaning rotate x for 80 degree, then rotate y 90 degree and no z rotatation
            # 6. A list of object location (x,y,z) w.r.t. glass. Ex: [13,-3,2] 13 units away from glass, 3 units to the left of glass, 2 units above
            
            # Emitters
            emitter_locations,
        ):
        for obj in objs_info:
            self.obj_dir.append(obj[0])
            if '.ply' in obj[0]:
                self.obj_type.append('ply')
            else:
                self.obj_type.append('obj')
            self.obj_bsdf.append(obj[1])
            self.obj_color.append(obj[2])
            self.obj_scale_factor.append(obj[3])
            self.obj_rotation.append(obj[4])
            self.obj_location.append(obj[5])
        
        self.emitter_locations = emitter_locations
        
        integrator = {
            'type': 'path',
            'max_depth': -1,
            'hide_emitters': True,
        }
        self.scene_dict['integrator'] = integrator
        self.create_emitters()
        self.create_sensors()
        self.create_obj()
        self.create_glass()

        scene = mi.load_dict(self.scene_dict)
        params = mi.traverse(scene)

        ref_images = []

        for sensor,angle in self.sensors:
            if abs(angle) <= self.max_viewing_angle/2:
                for i,emitter_name in enumerate(self.emitter_locations):
                    traverse_name = 'emitter' + str(i+1) + '.intensity.value'
                    params[traverse_name] = self.light_source_radiance(angle)
            else:
                for i,emitter_name in enumerate(self.emitter_locations):
                    traverse_name = 'emitter' + str(i+1) + '.intensity.value'
                    params[traverse_name] = 0.0
            ref_images.append(mi.render(scene, sensor=sensor, spp=self.sample_per_pixels))

        return ref_images
