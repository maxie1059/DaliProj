import matplotlib.pyplot as plt
import drjit as dr
import mitsuba as mi
import numpy as np
import cv2

mi.set_variant('cuda_ad_rgb')

to_world_sensor = mi.ScalarTransform4f().look_at(
    target=[0, 0, 0],
    origin=[15, -3, 0],
    up=[0, 0, 1]
)

integrator = {
    'type': 'path',
    'max_depth': -1,
    'hide_emitters': False,
}
emitter1 = {
    'type': 'point',
    'position': [3, 8, 6],
    'intensity': {
        'type': 'rgb',
        'value': [255,0,0]
    }
}
emitter2 = {
    'type': 'point',
    'position': [3, 8, 3],
    'intensity': {
        'type': 'rgb',
        'value': [0,255,0]
    }
}
emitter3 = {
    'type': 'point',
    'position': [3, 8, 0],
    'intensity': {
        'type': 'rgb',
        'value': [0,0,255]
    }
}
emitter4 = {
    'type': 'point',
    'position': [7, 0, 0],
    'intensity': {
        'type': 'rgb',
        'value': [100,100,100]
    }
}
sensor = {
    "type": "perspective",
        "fov_axis": "x",
        "fov": 34.0221,
        "to_world": to_world_sensor,
        # Sampler not actually used
        "sampler": {
            "type": "ldsampler",
            "sample_count": 1024
        },
        "film": {
            "type": "hdrfilm",
            "width": 1024,
            "height": 1024,
            "pixel_format": "rgb",
        },
}

obj = {
    'type': 'ply',
    'filename': '/mitsuba-tutorials/scenes/meshes/teapot.ply',
    'to_world': mi.ScalarTransform4f().translate([0, 0, -2]),
    'bsdf': {
        'type': 'diffuse',
        'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}, # obj color
    }
}

# Dust image pre-processing
image = cv2.imread('Dust-Textures-03.jpg',0).astype(float)
ret,normalizedImg = cv2.threshold(image,100,255,cv2.THRESH_TOZERO)
normalizedImg = cv2.normalize(normalizedImg, None, norm_type=cv2.NORM_MINMAX)
normalizedImg = mi.Bitmap(normalizedImg)

holographic_film = {
    'type': 'blendbsdf',
    'weight':{
        'type': 'bitmap',
        'bitmap': normalizedImg,
        'raw': True,
    },
    'bsdf_0': {
        'type': 'dielectric',
        'int_ior': 'bk7',
        'specular_reflectance': {
            'type': 'rgb',
            'value': [0.7, 0.7, 0.7]
        },
        'specular_transmittance': {
            'type': 'rgb',
            'value': [0.5, 0.7, 1.0]  # faint blue tint
        },
    },
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
    'sigma_t':{
        'type': 'rgb',
         'value': [0.1, 0.1, 0.1]  # Extinction coefficient
        },
        'albedo': {
            'type': 'rgb',
            'value': [0.1, 0.9, 0.9]  # Scattering albedo
        },
        'phase': {'type': 'isotropic'}
}
glass_slab = {
    "type": "cube",
    "to_world": mi.ScalarTransform4f().rotate(axis=(0, 1, 0), angle=90)
    @ mi.ScalarTransform4f().translate([0, 0, 6])
    @ mi.ScalarTransform4f().scale([2,4,0.05]),
    'bsdf': {'type': 'ref', 'id': 'holographic-film'},
    'interior': {'type': 'ref','id':'medium'}
}

scene_dict = {
    'type': 'scene',
    'integrator': integrator,
    'sensor': sensor,
    'teapot': obj,
    'light1': emitter1,
    'light2': emitter2,
    'light3': emitter3,
    'light4': emitter4,
    'holographic-film': holographic_film,
    'medium': medium,
    'slab': glass_slab,
}
scene = mi.load_dict(scene_dict)
img = mi.render(scene,spp = 128)
np_image = np.array(img)
plt.imshow(np_image** (1.0 / 2.2))
plt.axis('off')
plt.show()