import cv2
import os
import numpy as np
from PIL import Image, ImageChops
import imageio
from glob import glob
import matplotlib.pyplot as plt

def save_gif(vpath, images, fps=8):
    # imageio.mimsave(vpath, images, fps=fps, palettesize=256)
    images[0].save(vpath, save_all=True, append_images=images[3:80:3], duration=fps, loop=True)

def crop_mask(mask):
    if mask.dtype is not np.uint8:
        mask = mask.astype(np.uint8) * 255
    im = Image.fromarray(mask)
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, 0)
    bbox = diff.getbbox()
    return bbox


def crop_image_by_mask(img, mask):
    bbox = crop_mask(mask)
    try:
        crop_img = img.copy()[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    except:
        crop_img = img.copy()
    return crop_img

# vpath_list = ["/Users/xucao/Documents/MyPapers/cao2022ECCV/supp/data_run/Fig4_reading/iteration/draw_time_2022_03_11_19_25_52",
#               "/Users/xucao/Documents/MyPapers/cao2022ECCV/supp/data_run/Fig1_thinker/iteration/draw_time_2022_03_11_19_24_20",
#               "/Users/xucao/Documents/MyPapers/cao2022ECCV/supp/data_run/Fig6_bunny/iteration/draw_time_2022_03_11_19_33_02",]
#
# ndir_list = ["data/Fig4_reading",
#             "data/Fig1_thinker",
#             "data/Fig6_bunny"]
#
# gif_name = "teaser/synthetic.gif"

# vpath_list = ["/Users/xucao/Documents/MyPapers/cao2022ECCV/supp/data_run/Fig5_plant/iteration/draw_time_2022_03_11_19_30_58",
#               "/Users/xucao/Documents/MyPapers/cao2022ECCV/supp/data_run/Fig5_owl/iteration/draw_time_2022_07_25_03_47_55",
#               "/Users/xucao/Documents/MyPapers/cao2022ECCV/supp/data_run/Fig5_human/iteration/draw_time_2022_07_25_03_46_21",]
#
# ndir_list = ["data/Fig5_plant",
#              "data/Fig5_owl",
#              "data/Fig5_human"]
#
# gif_name = "teaser/real.gif"

vpath_list = ["/Users/xucao/Documents/MyPapers/cao2022ECCV/supp/data_run/supp_tent/iteration/draw_time_2022_03_11_19_49_15",
              "/Users/xucao/Documents/MyPapers/cao2022ECCV/supp/data_run/supp_vase/iteration/draw_time_2022_03_11_19_48_27"
              ]

ndir_list = ["data/supp_tent",
             "data/supp_vase"]

gif_name = "teaser/toy.gif"

# vpath_list = ["/Users/xucao/Documents/MyPapers/cao2022ECCV/supp/data_run/Fig7_diligent/bear/iteration/draw_time_2022_03_11_19_34_18",
#               # "/Users/xucao/Documents/MyPapers/cao2022ECCV/supp/data_run/Fig7_diligent/buddha/iteration/draw_time_2022_03_11_19_34_50",
#               "/Users/xucao/Documents/MyPapers/cao2022ECCV/supp/data_run/Fig7_diligent/cat/iteration/draw_time_2022_03_11_19_35_53",
#               "/Users/xucao/Documents/MyPapers/cao2022ECCV/supp/data_run/Fig7_diligent/cow/iteration/draw_time_2022_03_11_19_36_23"
#               ]
#
# ndir_list = ["data/Fig7_diligent/bear",
#              "data/Fig7_diligent/cat",
#              "data/Fig7_diligent/cow"]
#
# gif_name = "teaser/diligent1.gif"

# vpath_list = [
#               # "/Users/xucao/Documents/MyPapers/cao2022ECCV/supp/data_run/Fig7_diligent/buddha/iteration/draw_time_2022_03_11_19_34_50",
#               "/Users/xucao/Documents/MyPapers/cao2022ECCV/supp/data_run/Fig7_diligent/harvest/iteration/draw_time_2022_03_11_19_38_14",
#               "/Users/xucao/Documents/MyPapers/cao2022ECCV/supp/data_run/Fig7_diligent/pot1/iteration/draw_time_2022_03_11_19_39_31",
#               "/Users/xucao/Documents/MyPapers/cao2022ECCV/supp/data_run/Fig7_diligent/reading/iteration/draw_time_2022_03_11_19_41_10"
#               ]
#
# ndir_list = [
#             # "data/Fig7_diligent/buddha",
#              "data/Fig7_diligent/harvest",
#              "data/Fig7_diligent/pot1",
#             "data/Fig7_diligent/reading"
#             ]
#
# gif_name = "teaser/diligent2.gif"


img_height_list = []
img_width_list = []
for idx, vpath in enumerate(vpath_list):
    npath = os.path.join(ndir_list[idx], "normal_map.png")
    mask_path = os.path.join(ndir_list[idx], "mask.png")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(bool)
    normal_map = cv2.cvtColor(cv2.imread(npath, cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2BGR) / 256

    # plt.imshow(mask.astype(int))
    # plt.show()

    if "owl" in vpath:
        normal_map = normal_map.copy()[:230]
        mask = mask.copy()[:230]
    normal_map = crop_image_by_mask(normal_map, mask)

    iter_file_list = glob(os.path.join(vpath, "step*.jpeg"))
    iter_file_list.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    image = cv2.imread(iter_file_list[0])
    img_height, img_width, _ = image.shape
    n_height, n_width, _ = normal_map.shape
    img_height_list.append(img_height)
    img_height_list.append(n_height)
    img_width_list.append(img_width)
    img_width_list.append(n_width)

max_height = int(max(img_height_list) / 3)

font_scale = sum(img_width_list) / 5500
# font_scale = sum(img_width_list) / 6800
# font_thick = int(font_scale * 4)
font_thick = 2
pad_height = int(sum(img_width_list) / 100)

obj_frame_list = []
for idx, vpath in enumerate(vpath_list):
    npath = os.path.join(ndir_list[idx], "normal_map.png")
    mask_path = os.path.join(ndir_list[idx], "mask.png")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(bool)
    normal_map = cv2.cvtColor(cv2.imread(npath, cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2BGR)

    if normal_map.dtype is np.dtype(np.uint16):
        normal_map = normal_map/256

    if "owl" in vpath:
        normal_map = normal_map.copy()[:230]
        mask = mask.copy()[:230]
    normal_map[~mask] = 255

    normal_map = crop_image_by_mask(normal_map, mask)
    # plt.imshow(normal_map/256)
    # plt.show()
    n_height, n_width, _ = normal_map.shape
    n_width_scale = int(n_width * max_height / n_height)
    normal_map = cv2.resize(normal_map, (n_width_scale, max_height))

    normal_pad_img = np.ones((pad_height, n_width_scale, 3)) *255
    cv2.putText(normal_pad_img, f"Normal map",
                org=(int(n_width_scale/2)-85, int(pad_height/2)+5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=(0, 0, 0),
                thickness=font_thick
                )
    normal_map = np.concatenate((normal_map, normal_pad_img), axis=0)

    iter_file_list = glob(os.path.join(vpath, "step*.jpeg"))
    iter_file_list.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    frame_list = []
    for idx, img_path in enumerate(iter_file_list):
        image = cv2.imread(img_path)
        img_height, img_width, _ = image.shape
        img_width_scale = int(img_width * max_height / img_height)
        image = cv2.resize(image, (img_width_scale, max_height))

        # cv2.putText(image, f"Step {idx+1}",
        #             org=(0, 30),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=font_scale,
        #             color=(0, 0, 0),
        #             thickness=font_thick
        #             )

        image_pad_img = np.ones((pad_height, img_width_scale, 3))*255
        cv2.putText(image_pad_img, f"Surface: Step {idx+1}",
                org=(int(img_width_scale/2)-90, int(pad_height/2)+5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=(0, 0, 0),
                thickness=font_thick
                )

        image = np.concatenate((image, image_pad_img), axis=0)

        frame = np.concatenate((normal_map, np.ones((max_height+pad_height, 40, 3))*255, image), axis=1).astype(np.uint8)
        frame_list.append(frame)

    obj_frame_list.append(frame_list)

frame_num = [len(i) for i in obj_frame_list]
max_frame_num = max(frame_num)

for frame_list in obj_frame_list:
    last_frame = frame_list[-1].copy()
    while len(frame_list) < max_frame_num:
        frame_list.append(last_frame)

frame_save_list = []
for idx in range(max_frame_num):

    frame_save = np.concatenate([obj_frame_list[0][idx],
                                 np.ones((max_height+pad_height, 100, 3))*255,
                                 obj_frame_list[1][idx],
                                 # np.ones((max_height+pad_height, 100, 3))*255,
                                 # obj_frame_list[2][idx]
                                 ], axis=1).astype(np.uint8)
    frame_save_list.append(Image.fromarray(frame_save))

save_gif(gif_name, frame_save_list)
pass