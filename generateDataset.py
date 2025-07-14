from final_function import Hologram_Scene
import matplotlib.pyplot as plt
import mitsuba as mi
import os

mi.set_variant('cuda_ad_spectral')

root = './AllObjs'
directory = os.fsencode(root)
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    full_filepath = root + '/' + filename
    obj_name ,_ = os.path.splitext(filename)
    hologram_imgs = Hologram_Scene()
    obj_info = [full_filepath,'diffuse',[0.5,0.5,0.5],[1,1,1],[0,0,0],[8,0,0]]
    emitters_loc = [[-6, 8, 6],[-6, 8, -2]]
    hologram_imgs.number_of_cam = 3
    imgs = hologram_imgs.generate_hologram_imgs([obj_info],emitters_loc)
    for i,img in enumerate(imgs):
        img_name = './dataset/' + obj_name + str(i) + '.jpeg'
        pic = mi.util.convert_to_bitmap(img)
        pic.write(img_name)

    #comment out below if want to plot them
    # fig, axs = plt.subplots(1, hologram_imgs.number_of_cam, figsize=(15, 5))
    # for i in range(hologram_imgs.number_of_cam):
    #     axs.imshow(mi.util.convert_to_bitmap(img[i])) # add [i] after axs if there number_of_cam >1
    #     axs.axis('off')
# plt.show()