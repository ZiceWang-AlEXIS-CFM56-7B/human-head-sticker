# # USAGE
# # python example.py --source images/ocean_sunset.jpg --target images/ocean_day.jpg
#
# # import the necessary packages
# from color_transfer import color_transfer
# import numpy as np
# import argparse
# import cv2
#
# import matplotlib.pyplot as plt
#
# def show_image(title, image, width = 300):
# 	# resize the image to have a constant width, just to
# 	# make displaying the images take up less screen real
# 	# estate
# 	r = width / float(image.shape[1])
# 	dim = (width, int(image.shape[0] * r))
# 	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
#
# 	# show the resized image
# 	cv2.imshow(title, resized)
#
# def str2bool(v):
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')
#
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-s", "--source", required = True,
# 	help = "Path to the source image")
# ap.add_argument("-t", "--target", required = True,
# 	help = "Path to the target image")
# ap.add_argument("-co","--concat",required = True,
# 				help = "Path to the concat image")
# ap.add_argument("-c", "--clip", type = str2bool, default = 't',
# 	help = "Should np.clip scale L*a*b* values before final conversion to BGR? "
# 		   "Approptiate min-max scaling used if False.")
# ap.add_argument("-p", "--preservePaper", type = str2bool, default = 't',
# 	help = "Should color transfer strictly follow methodology layed out in original paper?")
# ap.add_argument("-o", "--output", help = "Path to the output image (optional)")
# args = vars(ap.parse_args())
#
# # load the images
# source = cv2.imread(args["source"])
# target = cv2.imread(args["target"])
#
# # transfer the color distribution from the source image
# # to the target image
# transfer = color_transfer(source, target, clip=args["clip"], preserve_paper=args["preservePaper"])
#
# # check to see if the output image should be saved
# if args["output"] is not None:
# 	cv2.imwrite(args["output"], transfer)
#
# # show the images and wait for a key press
# show_image("Source", source)
# show_image("Target", target)
# show_image("Transfer", transfer)
#
#
# cv2.waitKey(0)


from color_transfer import color_transfer
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob

# 主目录路径
base_dir = r"D:\work\project\VRC\color transfer\color-transfer0430\data+4"
# 获取主目录下的所有数字命名的子文件夹（例如：'1', '2', '3'...）
# subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f.isdigit()]
subfolders = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
debug = False
for folder in subfolders:
    imgpath = os.path.join(base_dir, folder)

    # 使用 glob 根据通配符查找对应的图像路径
    source_path = glob.glob(os.path.join(imgpath, "*maleuser*_skin*"))[0]
    target_path = glob.glob(os.path.join(imgpath, "*malemodel*_skin*"))[0]
    tcloth_path = glob.glob(os.path.join(imgpath, "*malemodel*noneskin*"))[0]
    ssmask_path = glob.glob(os.path.join(imgpath, "users*mask*"))[0]
    tsmask_path = glob.glob(os.path.join(imgpath, "models*mask*"))[0]
    tbmask_path = glob.glob(os.path.join(imgpath, "modelb*mask*"))[0]
    print([source_path,target_path])
    # 读取图像
    source = cv2.imread(source_path)
    target = cv2.imread(target_path)
    tcloth = cv2.imread(tcloth_path)
    ssmask = cv2.imread(ssmask_path, 0)  # 灰度图读取
    tsmask = cv2.imread(tsmask_path, 0)
    tbmask = cv2.imread(tbmask_path, 0)

    tsmask = 255*(tsmask > 128).astype(np.uint8)
    ssmask = 255*(ssmask > 128).astype(np.uint8)
    tbmask = 255*(tbmask > 128).astype(np.uint8)
    # Transfer the color distribution from the source image to the target image

    transfer = color_transfer(
        source=source,
        ssmask=ssmask,
        target=target,
        tsmask=tsmask,
        clip=True,
        preserve_paper=False,
        debug=debug
    )
    if debug:
        cv2.imshow("transfer", ssmask)
        cv2.waitKey(0)
    # Create result image by combining:
    # 1. Clothing from concat image (non-white regions)
    # 2. Color-transferred skin from transfer image (white regions of concat)
    result = tcloth.copy()
    result[tsmask>0] = transfer[tsmask>0]  # Use transferred skin where concat is white
    cv2.imwrite(os.path.join(imgpath, "outputnew.jpg"), result)

    #cv2.imshow("Result", result)
    #cv2.waitKey(0)